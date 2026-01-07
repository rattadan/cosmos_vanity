[![Build Status](https://github.com/hukkinj1/cosmosvanity/workflows/Tests/badge.svg?branch=master)](https://github.com/hukkinj1/cosmosvanity/actions?query=workflow%3ATests+branch%3Amaster+event%3Apush)
[![codecov.io](https://codecov.io/gh/hukkinj1/cosmosvanity/branch/master/graph/badge.svg)](https://codecov.io/gh/hukkinj1/cosmosvanity)
# cosmosvanity

<!--- Don't edit the version line below manually. Let bump2version do it for you. -->
> Version 1.0.0

> CLI tool for generating [Cosmos](https://cosmos.network) vanity addresses

## Features
* Generate Cosmos bech32 vanity addresses
* Use all CPU cores
* Specify a substring that the addresses must
    * start with
    * end with
    * contain
* Set required minimum amount of letters (a-z) or digits (0-9) in the addresses
* Binaries built for Linux, macOS and Windows

## Installing
Download the latest binary release from the [_Releases_](https://github.com/hukkinj1/cosmosvanity/releases) page. Alternatively, build from source yourself.

### Docker
You can also run cosmosvanity using Docker:

```bash
# Pull the image
docker pull rattadan/cosmosvanity

# Basic usage
docker run rattadan/cosmosvanity

# With specific parameters
docker run rattadan/cosmosvanity --startswith test
docker run rattadan/cosmosvanity --contains xyz --endswith 123

# Use multiple CPU cores (replace 4 with desired number of cores)
docker run --cpus=4 rattadan/cosmosvanity

# Generate multiple addresses
docker run rattadan/cosmosvanity -n 5
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
./cosmosvanity --mnemonic --startswith helloiamdavethebrave --leet
```

## Regex matching

You can also require the address payload (the part after `cosmos1`) to match a regular expression:

```bash
./cosmosvanity --regex "h3llo"
```

### Nerd-language regex (auto bech32-safe)

If you'd like to write patterns using common "nerd" substitutions (like `e`/`3`, `o`/`0`) while still staying compatible with the bech32 character set, you can use `--nerdregex`.

Example:

```bash
./cosmosvanity --nerdregex "h[e3]ll[o0]"
```
