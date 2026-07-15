[![Build Status](https://github.com/hukkinj1/cosmosvanity/workflows/Tests/badge.svg?branch=master)](https://github.com/hukkinj1/cosmosvanity/actions?query=workflow%3ATests+branch%3Amaster+event%3Apush)
[![codecov.io](https://codecov.io/gh/hukkinj1/cosmosvanity/branch/master/graph/badge.svg)](https://codecov.io/gh/hukkinj1/cosmosvanity)
# cosmosvanity

<!--- Don't edit the version line below manually. Let bump2version do it for you. -->
> Version 1.0.0

> CLI tool for generating [Cosmos](https://cosmos.network) vanity addresses

## Features
* Generate Cosmos bech32 vanity addresses
* Specify a substring that the addresses must
    * start with
    * end with
    * contain
* Set required minimum amount of letters (a-z) or digits (0-9) in the addresses

* New feature: Generate `ethsecp256k1` addresses for Cosmos EVM chains (like Evmos, Injective)
* Customize the bech32 prefix (e.g. `cosmos`, `evmos`, `inj`)


## Build from source (compile the binary)

```bash
go build -o cosmosvanity .
```

Run it:

```bash
./cosmosvanity --help
```

## Usage examples
Find an address that starts with "00000" (e.g. cosmos100000v3fpv4qg2a9ea6sj70gykxpt63wgjen2p)
```bash
./cosmosvanity --startswith 00000
```

Find an address that ends with "8888" (e.g. cosmos134dck5uddzjure8pyprmmqat96k3jlypn28888)
```bash
./cosmosvanity --endswith 8888
```

Find an address containing the substring "gener" (e.g. cosmos1z39wgener7azgh22s5a3pyswtnjkx2w0hvn3rv)
```bash
./cosmosvanity --contains gener
```

Find an address consisting of letters only (e.g. cosmos1rfqkejeaxlxwtjxucnrathlzgnvgcgldzmuxxe)
```bash
./cosmosvanity --letters 38
```

Find an address with at least 26 digits (e.g. cosmos1r573c4086585u084926726x535y3k2ktxpr88l)
```bash
./cosmosvanity --digits 26
```

Generate 5 addresses (the default is 1)
```bash
./cosmosvanity -n 5
```

Restrict to using only 1 CPU thread. This value defaults to the number of CPUs available.
```bash
./cosmosvanity --cpus 1
```

Combine flags introduced above
```bash
./cosmosvanity --contains 8888 --startswith a --endswith c
```

## EVM / ethsecp256k1 addresses

Generate an Evmos-style `ethsecp256k1` address (uses Ethereum address derivation and bech32 encoding):

```bash
./cosmosvanity --evm --prefix evmos --startswith 000
```

The same derivation works for any Cosmos EVM chain by changing the prefix, e.g. Injective:

```bash
./cosmosvanity --evm --prefix inj --startswith 000
```

## Custom bech32 prefix

You can change the bech32 prefix for non-EVM addresses too:

```bash
./cosmosvanity --prefix osmo --startswith 000
```

## Bech32-friendly encoding ("leet" mode)

Since Cosmos addresses use bech32, not every character you might want for a word is available (for example: `b`, `i`, `o`, and `1`).

This project includes a small encoder that converts normal text into a bech32-friendly pattern using simple leetspeak-style substitutions:

- **`o` -> `0`**
- **`i` -> `l`**
- **`b` -> `8`**
- **`1` -> `l`**

### Convert text to a bech32-friendly pattern

```bash
./cosmosvanity --encode helloiamdavethebrave
```

Example output:

```text
hell0lamdavethe8rave
```

### Search using normal words (auto-convert matchers)

When `--leet` is enabled, the values provided to `--startswith`, `--contains`, and `--endswith` are converted automatically before searching.

```bash
./cosmosvanity --startswith helloiamdavethebrave --leet
```

## Seed phrase (mnemonic)

Every generated address is derived from a random 12-word BIP39 seed phrase, which is printed alongside the address so you can recover it in any compatible wallet.

```bash
./cosmosvanity --startswith 000
```

## Regex matching

You can also require the address payload (the part after `cosmos1`) to match a regular expression:

```bash
./cosmosvanity --regex "h3llo"
```

### Nerd-language regex (auto bech32-safe)

Some letters are restricted in bech32 alphabet, so you can replace them with similiar symbols
If you'd like to write patterns using common "nerd" substitutions (like `e`/`3`, `o`/`0`) while still staying compatible with the bech32 character set, you can use `--nerdregex`.

Example:

```bash
./cosmosvanity --nerdregex "h[e3]ll[o0]"
```
