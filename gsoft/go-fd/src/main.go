package main

import (
	"bufio"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"golang.org/x/term"
)

const (
	yellow    = "\033[33m"
	gray      = "\033[90m"
	reset     = "\033[0m"
	eraseLine = "\033[2K\r"
)

func main() {
	pattern, startPath := parseArgs()
	re := compilePattern(pattern)
	ignorePatterns := loadIgnorePatterns()
	walkAndMatch(re, ignorePatterns, startPath)
}

func parseArgs() (string, string) {
	if len(os.Args) < 2 {
		fmt.Println("Usage: fd <regex> [start_path]")
		os.Exit(1)
	}
	pattern := os.Args[1]
	startPath := "/"
	if len(os.Args) >= 3 {
		startPath = os.Args[2]
	}
	return pattern, startPath
}

func compilePattern(pattern string) *regexp.Regexp {
	re, err := regexp.Compile(pattern)
	if err != nil {
		fmt.Printf("Invalid regex: %s\n", err)
		os.Exit(1)
	}
	return re
}

func loadIgnorePatterns() []*regexp.Regexp {
	exePath, err := os.Executable()
	if err != nil {
		return nil
	}
	ignoreFile := filepath.Join(filepath.Dir(exePath), ".ignore-fd.txt")
	f, err := os.Open(ignoreFile)
	if err != nil {
		return nil
	}
	defer f.Close()

	var patterns []*regexp.Regexp
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if re, err := regexp.Compile(line); err == nil {
			patterns = append(patterns, re)
		}
	}
	return patterns
}

func shouldIgnore(path string, patterns []*regexp.Regexp) bool {
	for _, p := range patterns {
		if p.MatchString(path) {
			return true
		}
	}
	return false
}

func walkAndMatch(re *regexp.Regexp, ignorePatterns []*regexp.Regexp, startPath string) {
	filepath.WalkDir(startPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return filepath.SkipDir
		}
		if !d.IsDir() {
			return nil
		}
		if shouldIgnore(path, ignorePatterns) {
			return filepath.SkipDir
		}
		showProgress(path)
		printIfMatch(re, path, d)
		return nil
	})
	clearProgress()
}

func showProgress(path string) {
	fmt.Printf("%s%s%s%s", eraseLine, gray, truncateToTermWidth(path), reset)
}

func truncateToTermWidth(s string) string {
	width, _, err := term.GetSize(int(os.Stdout.Fd()))
	if err != nil || width <= 0 {
		width = 80
	}
	maxLen := width - 1
	if len(s) > maxLen {
		return "..." + s[len(s)-(maxLen-3):]
	}
	return s
}

func clearProgress() {
	fmt.Print(eraseLine)
}

func printIfMatch(re *regexp.Regexp, path string, d fs.DirEntry) {
	name := d.Name()
	loc := re.FindStringIndex(name)
	if loc == nil {
		return
	}
	clearProgress()
	modTime := getModTime(d)
	highlighted := highlightMatch(name, loc)
	dir := filepath.Dir(path)
	fmt.Printf("%s/%s  %s\n", dir, highlighted, modTime)
}

func highlightMatch(name string, loc []int) string {
	return name[:loc[0]] + yellow + name[loc[0]:loc[1]] + reset + name[loc[1]:]
}

func getModTime(d fs.DirEntry) string {
	info, err := d.Info()
	if err != nil {
		return "unknown"
	}
	return info.ModTime().Format("2006-01-02 15:04:05")
}
