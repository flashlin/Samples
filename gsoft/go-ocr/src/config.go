package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
)

type Config struct {
	OCREndpoint        string `json:"ocr_endpoint"`
	TranslateEndpoint  string `json:"translate_endpoint"`
	TranslateModel     string `json:"translate_model"`
	TranslatePrompt    string `json:"translate_prompt"`
	ScreenshotHotkey   string `json:"screenshot_hotkey"`
	ClipboardOCRHotkey string `json:"clipboard_ocr_hotkey"`
	TranslateHotkey    string `json:"translate_hotkey"`
}

func DefaultConfig() Config {
	return Config{
		OCREndpoint:        "http://127.0.0.1:1240/ocr",
		TranslateEndpoint:  "http://127.0.0.1:11434/v1/chat/completions",
		TranslateModel:     "sun_leaf/HY-MT:1.8b",
		TranslatePrompt:    "Translate the following text to {target_lang}. Output ONLY the translation, with no explanation, no quotes, no markdown, no language label.",
		ScreenshotHotkey:   "shift+cmd+t",
		ClipboardOCRHotkey: "ctrl+cmd+t",
		TranslateHotkey:    "shift+cmd+r",
	}
}

func ConfigPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, "Library", "Application Support", "go-ocr", "config.json")
}

func LoadConfig() (*Config, error) {
	path := ConfigPath()
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		cfg := DefaultConfig()
		if err := SaveConfig(&cfg); err != nil {
			return nil, err
		}
		return &cfg, nil
	}
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	applyConfigDefaults(&cfg)
	return &cfg, nil
}

func SaveConfig(cfg *Config) error {
	path := ConfigPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func applyConfigDefaults(cfg *Config) {
	def := DefaultConfig()
	if cfg.OCREndpoint == "" {
		cfg.OCREndpoint = def.OCREndpoint
	}
	if cfg.TranslateEndpoint == "" {
		cfg.TranslateEndpoint = def.TranslateEndpoint
	}
	if cfg.ScreenshotHotkey == "" {
		cfg.ScreenshotHotkey = def.ScreenshotHotkey
	}
	if cfg.ClipboardOCRHotkey == "" {
		cfg.ClipboardOCRHotkey = def.ClipboardOCRHotkey
	}
	if cfg.TranslateModel == "" {
		cfg.TranslateModel = def.TranslateModel
	}
	if cfg.TranslatePrompt == "" {
		cfg.TranslatePrompt = def.TranslatePrompt
	}
	if cfg.TranslateHotkey == "" {
		cfg.TranslateHotkey = def.TranslateHotkey
	}
}
