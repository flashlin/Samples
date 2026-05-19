package main

import (
	"encoding/json"
	"strings"
	"unicode"
)

const (
	targetLangChinese = "Traditional Chinese (zh-TW)"
	targetLangEnglish = "English"
)

func RunTranslate(cfg *Config, sourceText string) (string, error) {
	target := detectTargetLanguage(sourceText)
	prompt := strings.ReplaceAll(cfg.TranslatePrompt, "{target_lang}", target)
	payload, err := buildTranslatePayload(cfg.TranslateModel, prompt, sourceText)
	if err != nil {
		return "", err
	}
	body, err := postOCR(cfg.OCREndpoint, payload)
	if err != nil {
		return "", err
	}
	return parseOCRResponse(body)
}

func detectTargetLanguage(text string) string {
	if isMostlyChinese(text) {
		return targetLangEnglish
	}
	return targetLangChinese
}

func isMostlyChinese(text string) bool {
	hanCount, latinCount := countHanAndLatin(text)
	return hanCount*4 > latinCount && hanCount > 0
}

func countHanAndLatin(text string) (int, int) {
	hanCount := 0
	latinCount := 0
	for _, r := range text {
		switch {
		case unicode.Is(unicode.Han, r):
			hanCount++
		case isAsciiLetter(r):
			latinCount++
		}
	}
	return hanCount, latinCount
}

func isAsciiLetter(r rune) bool {
	return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')
}

func buildTranslatePayload(model, prompt, text string) ([]byte, error) {
	keepAliveForever := -1
	payload := chatRequest{
		Model: model,
		Messages: []chatMessage{
			{
				Role: "user",
				Content: []contentPart{
					{Type: "text", Text: prompt + "\n\n" + text},
				},
			},
		},
		KeepAlive: &keepAliveForever,
	}
	return json.Marshal(payload)
}
