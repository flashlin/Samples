package main

import (
	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/widget"
)

type SettingsWindowDeps struct {
	App     fyne.App
	Config  *Config
	OnSaved func(*Config)
}

func OpenSettingsWindow(deps SettingsWindowDeps) {
	win := deps.App.NewWindow("go-ocr Settings")
	fields := newSettingsFields(deps.Config)
	form := buildSettingsForm(fields)
	buttons := buildSettingsButtons(win, fields, deps)

	win.SetContent(container.NewBorder(
		nil,
		buttons,
		nil,
		nil,
		container.NewVScroll(form),
	))
	win.Resize(fyne.NewSize(560, 520))
	win.Show()
}

type settingsFields struct {
	endpoint  *widget.Entry
	model     *widget.Entry
	prompt    *widget.Entry
	screenHK  *widget.Entry
	clipHK    *widget.Entry
}

func newSettingsFields(cfg *Config) *settingsFields {
	prompt := widget.NewMultiLineEntry()
	prompt.SetText(cfg.OCRPrompt)
	prompt.Wrapping = fyne.TextWrapWord
	prompt.SetMinRowsVisible(4)

	return &settingsFields{
		endpoint: entryWithValue(cfg.OCREndpoint),
		model:    entryWithValue(cfg.OCRModel),
		prompt:   prompt,
		screenHK: entryWithValue(cfg.ScreenshotHotkey),
		clipHK:   entryWithValue(cfg.ClipboardOCRHotkey),
	}
}

func entryWithValue(value string) *widget.Entry {
	e := widget.NewEntry()
	e.SetText(value)
	return e
}

func buildSettingsForm(f *settingsFields) fyne.CanvasObject {
	hkHint := "Format: modifier(s) + key, e.g. shift+cmd+t, ctrl+cmd+t"
	return container.NewVBox(
		widget.NewLabel("OCR Endpoint"),
		f.endpoint,
		widget.NewLabel("OCR Model"),
		f.model,
		widget.NewLabel("OCR Prompt"),
		f.prompt,
		widget.NewLabel("Screenshot Hotkey"),
		f.screenHK,
		widget.NewLabel("Clipboard OCR Hotkey"),
		f.clipHK,
		widget.NewLabelWithStyle(hkHint, fyne.TextAlignLeading, fyne.TextStyle{Italic: true}),
	)
}

func buildSettingsButtons(win fyne.Window, f *settingsFields, deps SettingsWindowDeps) fyne.CanvasObject {
	save := widget.NewButton("Save", func() { trySave(win, f, deps) })
	cancel := widget.NewButton("Cancel", func() { win.Close() })
	reset := widget.NewButton("Reset Defaults", func() { resetToDefaults(f) })
	return container.NewBorder(nil, nil, reset, container.NewHBox(cancel, save))
}

func resetToDefaults(f *settingsFields) {
	def := DefaultConfig()
	f.endpoint.SetText(def.OCREndpoint)
	f.model.SetText(def.OCRModel)
	f.prompt.SetText(def.OCRPrompt)
	f.screenHK.SetText(def.ScreenshotHotkey)
	f.clipHK.SetText(def.ClipboardOCRHotkey)
}

func trySave(win fyne.Window, f *settingsFields, deps SettingsWindowDeps) {
	newCfg := &Config{
		OCREndpoint:        f.endpoint.Text,
		OCRModel:           f.model.Text,
		OCRPrompt:          f.prompt.Text,
		ScreenshotHotkey:   f.screenHK.Text,
		ClipboardOCRHotkey: f.clipHK.Text,
	}
	if err := validateConfig(newCfg); err != nil {
		dialog.ShowError(err, win)
		return
	}
	if err := SaveConfig(newCfg); err != nil {
		dialog.ShowError(err, win)
		return
	}
	deps.OnSaved(newCfg)
	win.Close()
}

func validateConfig(cfg *Config) error {
	if _, err := ParseHotkey(cfg.ScreenshotHotkey); err != nil {
		return err
	}
	if _, err := ParseHotkey(cfg.ClipboardOCRHotkey); err != nil {
		return err
	}
	return nil
}
