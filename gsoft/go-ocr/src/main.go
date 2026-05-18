package main

import (
	"log"
	"os"
	"path/filepath"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
)

var version = "dev"

func main() {
	setupFileLogging()
	log.Printf("go-ocr %s starting", version)

	cfg := mustLoadConfig()
	SetCurrentConfig(cfg)
	mustInitClipboard()

	a := app.NewWithID("com.flash.go-ocr")
	a.SetIcon(trayIcon())

	hotkeys := NewHotkeyManager(RunScreenshotOCR, RunClipboardOCR)
	a.Lifecycle().SetOnStarted(func() {
		log.Printf("Fyne lifecycle started, registering hotkeys")
		registerHotkeysOrNotify(hotkeys, cfg)
		log.Printf("hotkey registration finished (screenshot=%s, clipboard=%s)",
			cfg.ScreenshotHotkey, cfg.ClipboardOCRHotkey)
	})

	if err := SetupTray(a, openSettingsHandler(a, hotkeys)); err != nil {
		log.Fatalf("setup tray: %v", err)
	}

	a.Run()
}

func setupFileLogging() {
	home, err := os.UserHomeDir()
	if err != nil {
		return
	}
	logDir := filepath.Join(home, "Library", "Logs")
	os.MkdirAll(logDir, 0o755)
	f, err := os.OpenFile(filepath.Join(logDir, "go-ocr.log"),
		os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return
	}
	log.SetOutput(f)
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
}

func mustLoadConfig() *Config {
	cfg, err := LoadConfig()
	if err != nil {
		log.Fatalf("load config: %v", err)
	}
	return cfg
}

func mustInitClipboard() {
	if err := InitClipboard(); err != nil {
		log.Fatalf("init clipboard: %v", err)
	}
}

func registerHotkeysOrNotify(m *HotkeyManager, cfg *Config) {
	if err := m.RegisterAll(cfg); err != nil {
		log.Printf("register hotkeys: %v", err)
		Notify("go-ocr", "Hotkey registration failed: "+err.Error())
	}
}

func openSettingsHandler(a fyne.App, hotkeys *HotkeyManager) func() {
	return func() {
		OpenSettingsWindow(SettingsWindowDeps{
			App:    a,
			Config: GetCurrentConfig(),
			OnSaved: func(newCfg *Config) {
				SetCurrentConfig(newCfg)
				if err := hotkeys.Reload(newCfg); err != nil {
					log.Printf("reload hotkeys: %v", err)
					Notify("go-ocr", "Hotkey reload failed: "+err.Error())
				}
			},
		})
	}
}
