package main

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"golang.org/x/term"
)

const (
	gray      = "\033[90m"
	reset     = "\033[0m"
	eraseLine = "\033[2K\r"
)

type sortMode int

const (
	sortDefault sortMode = iota
	sortByTime
	sortBySize
)

type dirEntry struct {
	path    string
	modTime time.Time
	size    int64
}

func main() {
	if len(os.Args) == 1 {
		entries := listTopLevel(".")
		sortEntries(entries, sortByTime)
		printEntries(entries)
		return
	}
	pattern, mode, ok := parseArgs(os.Args[1:])
	if !ok {
		printUsage()
		os.Exit(1)
	}
	entries := findMatches(".", pattern)
	sortEntries(entries, mode)
	printEntries(entries)
}

func parseArgs(args []string) (string, sortMode, bool) {
	mode := sortDefault
	var pattern string
	patternSet := false
	for _, a := range args {
		switch a {
		case "-?", "-h", "--help":
			return "", mode, false
		case "-d":
			mode = sortByTime
		case "-s":
			mode = sortBySize
		default:
			if strings.HasPrefix(a, "-") {
				return "", mode, false
			}
			if patternSet {
				return "", mode, false
			}
			pattern = a
			patternSet = true
		}
	}
	if !patternSet {
		pattern = ""
	}
	return pattern, mode, true
}

func printUsage() {
	fmt.Println("Usage: ld [options] <pattern>")
	fmt.Println("  Search subdirectories whose name contains <pattern>.")
	fmt.Println("Options:")
	fmt.Println("  -?, -h    Show this help")
	fmt.Println("  -d        Sort by modification time (newest first)")
	fmt.Println("  -s        Sort by disk size (largest first)")
	fmt.Println("  (default) Sort by mtime desc, then path length desc")
}

func listTopLevel(root string) []dirEntry {
	items, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	var results []dirEntry
	for _, d := range items {
		if !d.IsDir() {
			continue
		}
		showProgress(d.Name())
		info, err := d.Info()
		if err != nil {
			continue
		}
		results = append(results, dirEntry{
			path:    d.Name(),
			modTime: info.ModTime(),
			size:    dirSize(d.Name()),
		})
	}
	clearProgress()
	return results
}

func findMatches(root, pattern string) []dirEntry {
	var results []dirEntry
	filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return filepath.SkipDir
		}
		if !d.IsDir() {
			return nil
		}
		showProgress(path)
		if path == root {
			return nil
		}
		if !strings.Contains(d.Name(), pattern) {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		results = append(results, dirEntry{
			path:    path,
			modTime: info.ModTime(),
			size:    dirSize(path),
		})
		return nil
	})
	clearProgress()
	return results
}

func showProgress(path string) {
	if !term.IsTerminal(int(os.Stderr.Fd())) {
		return
	}
	fmt.Fprintf(os.Stderr, "%s%s%s%s", eraseLine, gray, truncateToTermWidth(path), reset)
}

func clearProgress() {
	if !term.IsTerminal(int(os.Stderr.Fd())) {
		return
	}
	fmt.Fprint(os.Stderr, eraseLine)
}

func truncateToTermWidth(s string) string {
	width, _, err := term.GetSize(int(os.Stderr.Fd()))
	if err != nil || width <= 0 {
		width = 80
	}
	maxLen := width - 1
	if len(s) > maxLen {
		return "..." + s[len(s)-(maxLen-3):]
	}
	return s
}

func dirSize(root string) int64 {
	var total int64
	filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		total += info.Size()
		return nil
	})
	return total
}

func sortEntries(entries []dirEntry, mode sortMode) {
	switch mode {
	case sortByTime:
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].modTime.After(entries[j].modTime)
		})
	case sortBySize:
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].size > entries[j].size
		})
	default:
		sort.Slice(entries, func(i, j int) bool {
			if !entries[i].modTime.Equal(entries[j].modTime) {
				return entries[i].modTime.After(entries[j].modTime)
			}
			return len(entries[i].path) > len(entries[j].path)
		})
	}
}

func printEntries(entries []dirEntry) {
	sizeWidth := 0
	for _, e := range entries {
		s := humanSize(e.size)
		if len(s) > sizeWidth {
			sizeWidth = len(s)
		}
	}
	for _, e := range entries {
		fmt.Printf("%*s  %s  %s\n",
			sizeWidth,
			humanSize(e.size),
			e.modTime.Format("2006-01-02 15:04:05"),
			e.path)
	}
}

func humanSize(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%dB", n)
	}
	div, exp := int64(unit), 0
	for x := n / unit; x >= unit; x /= unit {
		div *= unit
		exp++
	}
	units := []string{"K", "M", "G", "T", "P"}
	return fmt.Sprintf("%.1f%s", float64(n)/float64(div), units[exp])
}
