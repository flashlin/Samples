package main

import (
	"fmt"
	"os/exec"
	"strings"
)

func Notify(title, body string) {
	script := fmt.Sprintf(
		`display notification "%s" with title "%s"`,
		escapeAppleScript(body),
		escapeAppleScript(title),
	)
	exec.Command("osascript", "-e", script).Run()
}

func escapeAppleScript(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, `"`, `\"`)
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\r", " ")
	return s
}
