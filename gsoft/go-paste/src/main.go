package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const (
	typeImage = "image"
	typeFile  = "file"
	typeText  = "text"
)

func main() {
	clipType := detectClipboardType()
	switch clipType {
	case typeImage:
		handleImage()
	case typeFile:
		handleFile()
	case typeText:
		handleText()
	}
}

func detectClipboardType() string {
	output, err := runAppleScript("clipboard info")
	if err != nil || strings.TrimSpace(output) == "" {
		exitWithError("Clipboard is empty or not accessible")
		return ""
	}

	if containsAny(output, "PNGf", "TIFF") {
		return typeImage
	}
	if strings.Contains(output, "furl") {
		return typeFile
	}
	if containsAny(output, "utf8", "ut16", "«class utf8»", "«class ut16»") {
		return typeText
	}

	exitWithError("Unsupported clipboard content")
	return ""
}

func containsAny(s string, substrs ...string) bool {
	for _, sub := range substrs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

func handleImage() {
	filename := generateImageFilename()
	absPath := buildAbsPath(filename)

	script := fmt.Sprintf(`
try
	set pngData to the clipboard as «class PNGf»
on error
	set pngData to the clipboard as «class TIFF»
end try
set outFile to open for access (POSIX file "%s") with write permission
set eof outFile to 0
write pngData to outFile
close access outFile
`, absPath)

	_, err := runAppleScript(script)
	if err != nil {
		exitWithError("Failed to save image: %s", err)
	}
	fmt.Printf("Saved image to %s\n", filename)
}

func generateImageFilename() string {
	return fmt.Sprintf("paste_%s.png", time.Now().Format("20060102_150405"))
}

func buildAbsPath(filename string) string {
	dir, err := os.Getwd()
	if err != nil {
		exitWithError("Failed to get current directory: %s", err)
	}
	return filepath.Join(dir, filename)
}

func handleFile() {
	paths := getClipboardFilePaths()
	if len(paths) == 0 {
		exitWithError("No file paths found in clipboard")
	}

	printFileList(paths)
	action := askCopyOrMove()

	for _, src := range paths {
		executeFileAction(action, src)
	}
}

func getClipboardFilePaths() []string {
	script := `
use framework "AppKit"
set pb to current application's NSPasteboard's generalPasteboard()
set urls to pb's readObjectsForClasses:{current application's NSURL} options:(missing value)
if urls is missing value then return ""
set paths to {}
repeat with u in urls
	set end of paths to (u's |path|()) as text
end repeat
set AppleScript's text item delimiters to linefeed
return paths as text
`
	output, err := runAppleScript(script)
	if err != nil {
		exitWithError("Failed to get file paths: %s", err)
	}

	return filterEmptyLines(strings.Split(strings.TrimSpace(output), "\n"))
}

func filterEmptyLines(lines []string) []string {
	var result []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			result = append(result, trimmed)
		}
	}
	return result
}

func printFileList(paths []string) {
	fmt.Println("Files in clipboard:")
	for _, p := range paths {
		fmt.Printf("  %s\n", p)
	}
}

func askCopyOrMove() string {
	fmt.Print("Copy or Move? [c/m]: ")
	reader := bufio.NewReader(os.Stdin)
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(strings.ToLower(input))

	if input == "m" || input == "move" {
		return "move"
	}
	return "copy"
}

func executeFileAction(action string, src string) {
	name := filepath.Base(src)
	switch action {
	case "move":
		_, err := runCommand("mv", src, ".")
		if err != nil {
			fmt.Printf("Failed to move %s: %s\n", name, err)
			return
		}
		fmt.Printf("Moved %s\n", name)
	default:
		_, err := runCommand("cp", "-r", src, ".")
		if err != nil {
			fmt.Printf("Failed to copy %s: %s\n", name, err)
			return
		}
		fmt.Printf("Copied %s\n", name)
	}
}

func handleText() {
	output, err := runCommand("pbpaste")
	if err != nil {
		exitWithError("Failed to read clipboard text: %s", err)
	}
	fmt.Print(output)
}

func runCommand(name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("%s: %s", err, strings.TrimSpace(string(out)))
	}
	return string(out), nil
}

func runAppleScript(script string) (string, error) {
	return runCommand("osascript", "-e", script)
}

func exitWithError(format string, args ...any) {
	fmt.Printf(format+"\n", args...)
	os.Exit(1)
}
