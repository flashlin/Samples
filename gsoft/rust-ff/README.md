# File Finder

A Rust console application that searches for content within files using regular expressions, with filename filtering and colored output.

## Features

- Search file content using regular expressions
- Filter files by filename patterns using regular expressions
- Recursive search through directories and subdirectories
- Real-time progress display during search with terminal width awareness
- Colored output with highlighted matches
- Smart progress message truncation to prevent line wrapping
- Robust file handling (automatically skips binary and non-UTF-8 files)
- Cross-platform compatibility

## Usage

```bash
cargo run -- "<content_regex>" "<filename_regex>"
```

### Parameters

1. `content_regex`: Regular expression pattern to search within file content
2. `filename_regex`: Regular expression pattern to match filenames

### Examples

1. Find "error" or "warning" in all `.log` files:
```bash
cargo run -- "error|warning" ".*\.log$"
```

2. Find "TODO" or "FIXME" in all `.rs` files:
```bash
cargo run -- "TODO|FIXME" ".*\.rs$"
```

3. Find "main" function in Rust and TOML files:
```bash
cargo run -- "main" ".*\.(rs|toml)$"
```

4. Find "config" in all text files:
```bash
cargo run -- "config" ".*\.txt$"
```

5. Find any content in files starting with "test":
```bash
cargo run -- ".*" "^test.*"
```

## Output Format

The program displays:
- Real-time search progress (directories being searched)
  - Progress messages automatically truncate to fit terminal width
  - Long paths show only the end portion to prevent line wrapping
- File processing progress (current file being processed)
- Matches in format: `filename line_number content`
  - Filename in blue
  - Line number in yellow
  - Matching content highlighted in green
  - Other content in white

## Build

To build the project:
```bash
cargo build --release
```

The executable will be located at `target/release/ff`.

## Dependencies

- `regex`: For regular expression matching
- `walkdir`: For recursive directory traversal
- `colored`: For colored terminal output
- `terminal_size`: For detecting terminal width and smart progress display