package main

import (
	"errors"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/driver/desktop"
	"fyne.io/fyne/v2/theme"
)

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
	return theme.DocumentIcon()
}
