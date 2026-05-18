package main

import (
	"errors"
	"log"
	"os"
	"sync"
)

var (
	currentConfig    *Config
	currentConfigMu  sync.RWMutex
)

func SetCurrentConfig(cfg *Config) {
	currentConfigMu.Lock()
	currentConfig = cfg
	currentConfigMu.Unlock()
}

func GetCurrentConfig() *Config {
	currentConfigMu.RLock()
	defer currentConfigMu.RUnlock()
	return currentConfig
}

func RunScreenshotOCR() {
	log.Printf("RunScreenshotOCR: capturing region")
	path, err := CaptureRegion()
	if errors.Is(err, ErrScreenshotCancelled) {
		log.Printf("RunScreenshotOCR: cancelled by user")
		return
	}
	if errors.Is(err, ErrScreenRecordingDenied) {
		log.Printf("RunScreenshotOCR: permission denied")
		Notify("Screen Recording Permission Required", err.Error())
		return
	}
	if err != nil {
		log.Printf("RunScreenshotOCR: capture failed: %v", err)
		notifyFailure("Screenshot failed", err)
		return
	}
	defer os.Remove(path)

	data, err := os.ReadFile(path)
	if err != nil {
		log.Printf("RunScreenshotOCR: read file failed: %v", err)
		notifyFailure("Read screenshot", err)
		return
	}
	log.Printf("RunScreenshotOCR: got %d bytes from %s", len(data), path)
	WriteClipboardImage(data)
	log.Printf("RunScreenshotOCR: image written to clipboard")
	Notify("go-ocr", "Screenshot captured, running OCR...")
	runOCRAndNotify(data)
}

func RunClipboardOCR() {
	log.Printf("RunClipboardOCR: reading clipboard image")
	data, err := ReadClipboardImage()
	if err != nil {
		log.Printf("RunClipboardOCR: no image: %v", err)
		Notify("OCR Failed", "No image in clipboard")
		return
	}
	log.Printf("RunClipboardOCR: got %d bytes", len(data))
	runOCRAndNotify(data)
}

func runOCRAndNotify(png []byte) {
	cfg := GetCurrentConfig()
	log.Printf("OCR: POST %s model=%s bytes=%d", cfg.OCREndpoint, cfg.OCRModel, len(png))
	text, err := RunOCR(OCRRequest{Config: cfg, PNG: png})
	if err != nil {
		log.Printf("OCR: failed: %v", err)
		notifyFailure("OCR Failed", err)
		return
	}
	log.Printf("OCR: got %d chars of text", len(text))
	WriteClipboardText(text)
	Notify("OCR Success", firstLine(text))
}

func notifyFailure(title string, err error) {
	log.Printf("%s: %v", title, err)
	Notify(title, truncate(err.Error(), 200))
}

func firstLine(s string) string {
	for i, r := range s {
		if r == '\n' {
			return truncate(s[:i], 120)
		}
	}
	return truncate(s, 120)
}
