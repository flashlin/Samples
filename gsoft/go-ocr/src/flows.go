package main

import (
	"errors"
	"log"
	"os"
	"strings"
	"sync"
)

var (
	currentConfig    *Config
	currentConfigMu  sync.RWMutex
	inFlightMu       sync.Mutex
)

func tryAcquireInFlight() bool {
	return inFlightMu.TryLock()
}

func releaseInFlight() {
	inFlightMu.Unlock()
}

func notifyBusy() {
	Notify("go-ocr", "Busy, please wait for current request to finish")
}

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
	if !tryAcquireInFlight() {
		log.Printf("RunScreenshotOCR: busy, skipping")
		notifyBusy()
		return
	}
	defer releaseInFlight()
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
	if !tryAcquireInFlight() {
		log.Printf("RunClipboardOCR: busy, skipping")
		notifyBusy()
		return
	}
	defer releaseInFlight()
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

func RunClipboardTranslate() {
	if !tryAcquireInFlight() {
		log.Printf("RunClipboardTranslate: busy, skipping")
		notifyBusy()
		return
	}
	defer releaseInFlight()
	Notify("go-ocr", "Starting translation...")
	text, err := resolveTranslateSourceText()
	if err != nil {
		log.Printf("RunClipboardTranslate: no source: %v", err)
		Notify("Translate Failed", "No image or text in clipboard")
		return
	}
	runTranslateAndNotify(text)
}

func resolveTranslateSourceText() (string, error) {
	if text := strings.TrimSpace(ReadClipboardText()); text != "" {
		log.Printf("RunClipboardTranslate: using clipboard text (%d chars)", len(text))
		return text, nil
	}
	log.Printf("RunClipboardTranslate: no text, trying image")
	data, err := ReadClipboardImage()
	if err != nil {
		return "", err
	}
	log.Printf("RunClipboardTranslate: got %d image bytes, running OCR", len(data))
	cfg := GetCurrentConfig()
	ocrText, err := RunOCR(OCRRequest{Config: cfg, PNG: data})
	if err != nil {
		notifyFailure("OCR Failed", err)
		return "", err
	}
	WriteClipboardText(ocrText)
	Notify("go-ocr", "OCR done, translating...")
	return ocrText, nil
}

func runTranslateAndNotify(source string) {
	cfg := GetCurrentConfig()
	log.Printf("Translate: POST %s model=%s chars=%d", cfg.OCREndpoint, cfg.TranslateModel, len(source))
	translated, err := RunTranslate(cfg, source)
	if err != nil {
		log.Printf("Translate: failed: %v", err)
		notifyFailure("Translate Failed", err)
		return
	}
	log.Printf("Translate: got %d chars", len(translated))
	WriteClipboardText(translated)
	Notify("Translate Success", firstLine(translated))
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
