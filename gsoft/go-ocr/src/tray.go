package main

import (
	_ "embed"
	"errors"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/driver/desktop"
	"fyne.io/fyne/v2/theme"
)

//go:embed icon.png
var iconPNG []byte

func SetupTray(a fyne.App, onOpenSettings func()) error {
	desk, ok := a.(desktop.App)
	if !ok {
		return errors.New("system tray not supported on this platform")
	}
	menu := fyne.NewMenu("go-ocr",
		fyne.NewMenuItem("Setting", onOpenSettings),
		fyne.NewMenuItem("Quit", func() { a.Quit() }),
	)
	desk.SetSystemTrayMenu(menu)
	desk.SetSystemTrayIcon(trayIcon())
	return nil
}

func trayIcon() fyne.Resource {
	if len(iconPNG) > 0 {
		return fyne.NewStaticResource("icon.png", iconPNG)
	}
	return theme.DocumentIcon()
}
