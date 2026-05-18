package main

import (
	"fmt"
	"log"
	"os/exec"
	"strings"
)

func Notify(title, body string) {
	script := fmt.Sprintf(
		`display notification "%s" with title "%s"`,
		escapeAppleScript(body),
		escapeAppleScript(title),
	)
	cmd := exec.Command("osascript", "-e", script)
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Notify failed: %v output=%q title=%q body=%q",
			err, string(output), title, body)
		return
	}
	log.Printf("Notify sent: title=%q body=%q", title, truncate(body, 80))
}

func escapeAppleScript(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, `"`, `\"`)
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\r", " ")
	return s
}
