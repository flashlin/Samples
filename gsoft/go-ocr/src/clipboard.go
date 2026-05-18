package main

import (
	"errors"

	"golang.design/x/clipboard"
)

var ErrNoClipboardImage = errors.New("no image in clipboard")

func InitClipboard() error {
	return clipboard.Init()
}

func ReadClipboardImage() ([]byte, error) {
	data := clipboard.Read(clipboard.FmtImage)
	if len(data) == 0 {
		return nil, ErrNoClipboardImage
	}
	return data, nil
}

func WriteClipboardText(s string) {
	clipboard.Write(clipboard.FmtText, []byte(s))
}

func WriteClipboardImage(png []byte) {
	clipboard.Write(clipboard.FmtImage, png)
}
