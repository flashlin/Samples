package main

import (
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

var (
	ErrScreenshotCancelled = errors.New("screenshot cancelled")
	ErrScreenRecordingDenied = errors.New("Screen Recording permission denied. " +
		"Grant access in System Settings -> Privacy & Security -> Screen Recording, then restart go-ocr.")
)

func CaptureRegion() (string, error) {
	path := newScreenshotPath()
	cmd := exec.Command("screencapture", "-i", "-s", "-x", path)
	output, runErr := cmd.CombinedOutput()
	if runErr != nil {
		log.Printf("screencapture exit=%v output=%q", runErr, string(output))
	}
	if screenshotProduced(path) {
		return path, nil
	}
	os.Remove(path)
	if isPermissionDenied(string(output)) {
		return "", ErrScreenRecordingDenied
	}
	return "", ErrScreenshotCancelled
}

func isPermissionDenied(stderr string) bool {
	s := strings.ToLower(stderr)
	return strings.Contains(s, "could not create image from rect") ||
		strings.Contains(s, "not authorized") ||
		strings.Contains(s, "tcc")
}

func newScreenshotPath() string {
	name := fmt.Sprintf("go-ocr-%d.png", time.Now().UnixNano())
	return filepath.Join(os.TempDir(), name)
}

func screenshotProduced(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return info.Size() > 0
}
