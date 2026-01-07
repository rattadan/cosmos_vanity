package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"regexp"
	"runtime"
	"strings"
	"sync/atomic"
	"time"
	"unicode"

	flag "github.com/spf13/pflag"

	"github.com/cosmos/cosmos-sdk/types/bech32"
	"github.com/cosmos/go-bip39"
	"github.com/tendermint/tendermint/crypto/secp256k1"
)

type matcher struct {
	StartsWith string
	EndsWith   string
	Contains   string
	Regex      string
	Letters    int
	Digits     int
}

func (m matcher) Match(candidate string) bool {
	candidate = strings.TrimPrefix(candidate, "cosmos1")
	if !strings.HasPrefix(candidate, m.StartsWith) {
		return false
	}
	if !strings.HasSuffix(candidate, m.EndsWith) {
		return false
	}
	if !strings.Contains(candidate, m.Contains) {
		return false
	}
	if m.Regex != "" {
		re, err := regexp.Compile(m.Regex)
		if err != nil {
			return false
		}
		if !re.MatchString(candidate) {
			return false
		}
	}
	if countUnionChars(candidate, bech32digits) < m.Digits {
		return false
	}
	if countUnionChars(candidate, bech32letters) < m.Letters {
		return false
	}
	return true
}

func (m matcher) ValidationErrors() []string {
	var errs []string
	if !bech32Only(m.Contains) || !bech32Only(m.StartsWith) || !bech32Only(m.EndsWith) {
		errs = append(errs, "ERROR: A provided matcher contains bech32 incompatible characters")
	}
	if len(m.Contains) > 38 || len(m.StartsWith) > 38 || len(m.EndsWith) > 38 {
		errs = append(errs, "ERROR: A provided matcher is too long. Must be max 38 characters.")
	}
	if m.Digits < 0 || m.Letters < 0 {
		errs = append(errs, "ERROR: Can't require negative amount of characters")
	}
	if m.Digits+m.Letters > 38 {
		errs = append(errs, "ERROR: Can't require more than 38 characters")
	}
	if m.Regex != "" {
		if _, err := regexp.Compile(m.Regex); err != nil {
			errs = append(errs, fmt.Sprintf("ERROR: Invalid regex pattern: %v", err))
		}
	}
	return errs
}

type wallet struct {
	Address string
	Pubkey  []byte
	Privkey []byte
}

func (w wallet) String() string {
	return "Address:\t" + w.Address + "\n" +
		"Public key:\t" + hex.EncodeToString(w.Pubkey) + "\n" +
		"Private key:\t" + hex.EncodeToString(w.Privkey)
}

// GenerateMnemonic creates a new 12-word mnemonic seed phrase
func GenerateMnemonic() (string, error) {
	// Generate 128 bits of entropy (for 12 words)
	entropy, err := bip39.NewEntropy(128)
	if err != nil {
		return "", err
	}

	mnemonic, err := bip39.NewMnemonic(entropy)
	if err != nil {
		return "", err
	}

	return mnemonic, nil
}

// GenerateWalletFromMnemonic creates a wallet from a 12-word mnemonic
func GenerateWalletFromMnemonic(mnemonic string) (wallet, string, error) {
	// Validate the mnemonic first
	if !bip39.IsMnemonicValid(mnemonic) {
		return wallet{}, "", fmt.Errorf("invalid mnemonic")
	}

	// Convert mnemonic to seed
	seed := bip39.NewSeed(mnemonic, "") // Using empty password for now

	// Derive a key from the seed using a simple approach
	derivedKey := sha256.Sum256(seed)

	// Use the first 32 bytes for private key
	privKeyBytes := derivedKey[:32]

	// Create a new private key by copying the derived bytes
	privKey := secp256k1.GenPrivKeySecp256k1(privKeyBytes)
	pubKey := privKey.PubKey().(secp256k1.PubKey)

	bech32Addr, err := bech32.ConvertAndEncode("cosmos", pubKey.Address())
	if err != nil {
		return wallet{}, "", err
	}

	return wallet{bech32Addr, pubKey, privKey}, mnemonic, nil
}

func generateWallet() wallet {
	var privkey secp256k1.PrivKey = secp256k1.GenPrivKey()
	var pubkey secp256k1.PubKey = privkey.PubKey().(secp256k1.PubKey)
	bech32Addr, err := bech32.ConvertAndEncode("cosmos", pubkey.Address())
	if err != nil {
		panic(err)
	}

	return wallet{bech32Addr, pubkey, privkey}
}

func findMatchingWallets(ch chan wallet, quit chan struct{}, m matcher, attempts *uint64) {
	for {
		select {
		case <-quit:
			return
		default:
			w := generateWallet()
			atomic.AddUint64(attempts, 1)
			if m.Match(w.Address) {
				select {
				case ch <- w:
				default:
				}
			}
		}
	}
}

// walletWithMnemonic holds both the wallet and its associated mnemonic
type walletWithMnemonic struct {
	Wallet   wallet
	Mnemonic string
}

// findMatchingWalletsMnemonic finds wallets using 12-word mnemonic generation
func findMatchingWalletsMnemonic(ch chan walletWithMnemonic, quit chan struct{}, m matcher, attempts *uint64) {
	for {
		select {
		case <-quit:
			return
		default:
			mnemonic, err := GenerateMnemonic()
			if err != nil {
				continue // Skip on error
			}

			w, usedMnemonic, err := GenerateWalletFromMnemonic(mnemonic)
			if err != nil {
				continue // Skip on error
			}

			atomic.AddUint64(attempts, 1)
			if m.Match(w.Address) {
				select {
				case ch <- walletWithMnemonic{w, usedMnemonic}:
				default:
				}
			}
		}
	}
}

// findMatchingWalletMnemonic finds wallets using mnemonic-based generation
func findMatchingWalletMnemonic(m matcher, goroutines int) (wallet, string) {
	ch := make(chan walletWithMnemonic)
	quit := make(chan struct{})
	defer close(quit)

	var attempts uint64
	expectedAttempts := m.calculateExpectedAttempts()

	// Start monitoring hash rate
	go monitorHashRate(&attempts, quit, expectedAttempts)

	// Start mnemonic-based worker goroutines
	for i := 0; i < goroutines; i++ {
		go findMatchingWalletsMnemonic(ch, quit, m, &attempts)
	}

	result := <-ch
	return result.Wallet, result.Mnemonic
}

func (m matcher) calculateExpectedAttempts() float64 {
	// Base probability space for bech32 characters
	const addressLength = 38 // Standard cosmos address length after prefix

	// Start with base probability
	probability := 1.0

	// Calculate probability for StartsWith
	if len(m.StartsWith) > 0 {
		// Each position has 1/32 chance for exact match
		probability *= math.Pow(1.0/32.0, float64(len(m.StartsWith)))
	}

	// Calculate probability for EndsWith
	if len(m.EndsWith) > 0 {
		// Each position has 1/32 chance for exact match
		probability *= math.Pow(1.0/32.0, float64(len(m.EndsWith)))
	}

	// Calculate probability for Contains
	if len(m.Contains) > 0 {
		// For contains, we have multiple possible positions
		// Probability is higher than exact match but still requires all characters
		containsLen := float64(len(m.Contains))
		possiblePositions := float64(addressLength - len(m.Contains) + 1)
		probability *= (possiblePositions * math.Pow(1.0/32.0, containsLen))
	}

	// Calculate probability for required letters and digits
	if m.Letters > 0 {
		// Probability of getting a letter in one position
		pLetter := float64(len(bech32letters)) / 32.0
		probability *= math.Pow(pLetter, float64(m.Letters))
	}

	if m.Digits > 0 {
		// Probability of getting a digit in one position
		pDigit := float64(len(bech32digits)) / 32.0
		probability *= math.Pow(pDigit, float64(m.Digits))
	}

	// Return expected number of attempts (1/probability)
	if probability > 0 {
		return 1.0 / probability
	}
	return math.MaxFloat64
}

func formatDuration(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%.1fs", d.Seconds())
	} else if d < time.Hour {
		return fmt.Sprintf("%dm%ds", int(d.Minutes()), int(d.Seconds())%60)
	} else {
		return fmt.Sprintf("%dh%dm", int(d.Hours()), int(d.Minutes())%60)
	}
}

func monitorHashRate(attempts *uint64, quit chan struct{}, expectedAttempts float64) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	var lastCount uint64
	var lastTime time.Time = time.Now()
	startTime := time.Now()

	for {
		select {
		case <-quit:
			return
		case <-ticker.C:
			currentCount := atomic.LoadUint64(attempts)
			currentTime := time.Now()

			hashRate := float64(currentCount-lastCount) / currentTime.Sub(lastTime).Seconds()

			// Calculate ETA
			if hashRate > 0 {
				remainingAttempts := expectedAttempts - float64(currentCount)
				if remainingAttempts < 0 {
					remainingAttempts = 0
				}
				etaSeconds := remainingAttempts / hashRate
				eta := formatDuration(time.Duration(etaSeconds) * time.Second)
				elapsedTime := formatDuration(time.Since(startTime))

				fmt.Printf("\rHash rate: %.2f h/s | Total hashes: %d | Elapsed: %s | ETA: %s",
					hashRate, currentCount, elapsedTime, eta)
			}

			lastCount = currentCount
			lastTime = currentTime
		}
	}
}

func findMatchingWalletConcurrent(m matcher, goroutines int, useCUDA bool) wallet {
	ch := make(chan wallet)
	quit := make(chan struct{})
	defer close(quit)

	var attempts uint64
	expectedAttempts := m.calculateExpectedAttempts()

	// Start monitoring hash rate
	go monitorHashRate(&attempts, quit, expectedAttempts)

	// Start CPU worker goroutines if requested
	if goroutines > 0 {
		for i := 0; i < goroutines; i++ {
			go findMatchingWallets(ch, quit, m, &attempts)
		}
	}

	// If CUDA is available and enabled, start CUDA worker in parallel
	if useCUDA && CUDAAvailable() {
		go func() {
			batchSize := 100000 // Process 100k addresses at once
			for {
				select {
				case <-quit:
					return
				default:
					wallets := GenerateWalletsCUDA(batchSize)
					atomic.AddUint64(&attempts, uint64(batchSize))
					for _, w := range wallets {
						if m.Match(w.Address) {
							select {
							case ch <- w:
							default:
							}
							return
						}
					}
				}
			}
		}()
	}

	return <-ch
}

const bech32digits = "023456789"
const bech32letters = "acdefghjklmnpqrstuvwxyzACDEFGHJKLMNPQRSTUVWXYZ"

// This is alphanumeric chars minus chars "1", "b", "i", "o" (case insensitive)
const bech32chars = bech32digits + bech32letters

func bech32Only(s string) bool {
	return countUnionChars(s, bech32chars) == len(s)
}

func countUnionChars(s string, letterSet string) int {
	count := 0
	for _, c := range s {
		if strings.ContainsRune(letterSet, c) {
			count++
		}
	}
	return count
}

func toBech32Friendly(s string) string {
	s = strings.ToLower(s)
	var b strings.Builder
	b.Grow(len(s))

	for _, r := range s {
		switch r {
		case '0', '2', '3', '4', '5', '6', '7', '8', '9':
			b.WriteRune(r)
		case '1':
			b.WriteRune('l')
		case 'a', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z':
			b.WriteRune(r)
		case 'b':
			b.WriteRune('8')
		case 'i':
			b.WriteRune('l')
		case 'o':
			b.WriteRune('0')
		default:
			switch r {
			case '@':
				b.WriteRune('a')
			case '$':
				b.WriteRune('s')
			case '!':
				b.WriteRune('l')
			}
		}
	}

	return b.String()
}

func mapRuneToBech32Friendly(r rune) (rune, bool) {
	r = unicode.ToLower(r)
	switch r {
	case '0', '2', '3', '4', '5', '6', '7', '8', '9':
		return r, true
	case '1':
		return 'l', true
	case 'a', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z':
		return r, true
	case 'b':
		return '8', true
	case 'i':
		return 'l', true
	case 'o':
		return '0', true
	default:
		return 0, false
	}
}

func nerdRegexToBech32Safe(pattern string) string {
	pattern = strings.ToLower(pattern)
	var b strings.Builder
	b.Grow(len(pattern))

	inClass := false
	escaped := false
	for _, r := range pattern {
		if escaped {
			b.WriteRune('\\')
			if mr, ok := mapRuneToBech32Friendly(r); ok {
				b.WriteRune(mr)
			} else {
				b.WriteRune(r)
			}
			escaped = false
			continue
		}

		if r == '\\' {
			escaped = true
			continue
		}

		if inClass {
			switch r {
			case ']':
				inClass = false
				b.WriteRune(r)
			case '-', '^':
				b.WriteRune(r)
			default:
				if mr, ok := mapRuneToBech32Friendly(r); ok {
					b.WriteRune(mr)
				}
			}
			continue
		}

		switch r {
		case '[':
			inClass = true
			b.WriteRune(r)
		case '.', '*', '+', '?', '{', '}', '(', ')', '|', '^', '$':
			b.WriteRune(r)
		default:
			if mr, ok := mapRuneToBech32Friendly(r); ok {
				b.WriteRune(mr)
			}
		}
	}

	if escaped {
		b.WriteRune('\\')
	}

	return b.String()
}

func main() {
	var walletsToFind = flag.IntP("count", "n", 1, "Amount of matching wallets to find")
	var cpuCount = flag.Int("cpus", runtime.NumCPU(), "Amount of CPU cores to use")
	var useCUDA = flag.Bool("cuda", false, "Use CUDA acceleration if available")
	var useMnemonic = flag.Bool("mnemonic", false, "Generate addresses from random 12-word mnemonics")
	var useLeet = flag.Bool("leet", false, "Convert provided matchers (startswith/contains/endswith) to bech32-friendly leetspeak")
	var encodeOnly = flag.String("encode", "", "Convert arbitrary text to a bech32-friendly pattern and exit")

	var mustContain = flag.StringP("contains", "c", "", "A string that the address must contain")
	var mustStartWith = flag.StringP("startswith", "s", "", "A string that the address must start with")
	var mustEndWith = flag.StringP("endswith", "e", "", "A string that the address must end with")
	var regexPattern = flag.StringP("regex", "r", "", "A regex pattern the address must match (e.g., 'h3llo' for leetspeak)")
	var nerdRegexPattern = flag.String("nerdregex", "", "A nerd-language regex pattern (e.g., 'h[e3]ll[o0]') that will be converted to a bech32-safe regex")
	var letters = flag.IntP("letters", "l", 0, "Amount of letters (a-z) that the address must contain")
	var digits = flag.IntP("digits", "d", 0, "Amount of digits (0-9) that the address must contain")
	flag.Parse()

	if *encodeOnly != "" {
		fmt.Println(toBech32Friendly(*encodeOnly))
		return
	}

	if *walletsToFind < 1 {
		fmt.Println("ERROR: The number of wallets to generate must be 1 or more")
		os.Exit(1)
	}
	if *regexPattern != "" && *nerdRegexPattern != "" {
		fmt.Println("ERROR: Can't use both --regex and --nerdregex")
		os.Exit(1)
	}
	if *cpuCount < 0 || (*cpuCount == 0 && !*useCUDA) {
		fmt.Println("ERROR: Must use either CUDA or at least 1 CPU core")
		os.Exit(1)
	}

	startsWith := strings.ToLower(*mustStartWith)
	endsWith := strings.ToLower(*mustEndWith)
	contains := strings.ToLower(*mustContain)
	if *useLeet {
		startsWith = toBech32Friendly(startsWith)
		endsWith = toBech32Friendly(endsWith)
		contains = toBech32Friendly(contains)
	}

	regex := *regexPattern
	if *nerdRegexPattern != "" {
		regex = nerdRegexToBech32Safe(*nerdRegexPattern)
	}

	m := matcher{
		StartsWith: startsWith,
		EndsWith:   endsWith,
		Contains:   contains,
		Regex:      regex,
		Letters:    *letters,
		Digits:     *digits,
	}
	matcherValidationErrs := m.ValidationErrors()
	if len(matcherValidationErrs) > 0 {
		for i := 0; i < len(matcherValidationErrs); i++ {
			fmt.Println(matcherValidationErrs[i])
		}
		os.Exit(1)
	}

	var matchingWallet wallet
	var mnemonic string

	for i := 0; i < *walletsToFind; i++ {
		if *useMnemonic {
			matchingWallet, mnemonic = findMatchingWalletMnemonic(m, *cpuCount)
		} else {
			matchingWallet = findMatchingWalletConcurrent(m, *cpuCount, *useCUDA)
		}

		fmt.Printf(":::: Matching wallet %d/%d found ::::\n", i+1, *walletsToFind)
		fmt.Println(matchingWallet)

		if *useMnemonic && mnemonic != "" {
			fmt.Printf("Mnemonic:\t%s\n", mnemonic)
		}
	}
}
