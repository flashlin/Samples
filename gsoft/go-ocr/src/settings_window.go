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
	endpoint        *widget.Entry
	model           *widget.Entry
	prompt          *widget.Entry
	translateModel  *widget.Entry
	translatePrompt *widget.Entry
	screenHK        *widget.Entry
	clipHK          *widget.Entry
	translateHK     *widget.Entry
}

func newSettingsFields(cfg *Config) *settingsFields {
	prompt := newMultiLineEntry(cfg.OCRPrompt)
	translatePrompt := newMultiLineEntry(cfg.TranslatePrompt)

	return &settingsFields{
		endpoint:        entryWithValue(cfg.OCREndpoint),
		model:           entryWithValue(cfg.OCRModel),
		prompt:          prompt,
		translateModel:  entryWithValue(cfg.TranslateModel),
		translatePrompt: translatePrompt,
		screenHK:        entryWithValue(cfg.ScreenshotHotkey),
		clipHK:          entryWithValue(cfg.ClipboardOCRHotkey),
		translateHK:     entryWithValue(cfg.TranslateHotkey),
	}
}

func newMultiLineEntry(value string) *widget.Entry {
	e := widget.NewMultiLineEntry()
	e.SetText(value)
	e.Wrapping = fyne.TextWrapWord
	e.SetMinRowsVisible(4)
	return e
}

func entryWithValue(value string) *widget.Entry {
	e := widget.NewEntry()
	e.SetText(value)
	return e
}

func buildSettingsForm(f *settingsFields) fyne.CanvasObject {
	hkHint := "Format: modifier(s) + key, e.g. shift+cmd+t, ctrl+cmd+t"
	tpHint := "Use {target_lang} placeholder; replaced with English or Traditional Chinese (zh-TW)"
	return container.NewVBox(
		widget.NewLabel("OCR Endpoint"),
		f.endpoint,
		widget.NewLabel("OCR Model"),
		f.model,
		widget.NewLabel("OCR Prompt"),
		f.prompt,
		widget.NewLabel("Translate Model"),
		f.translateModel,
		widget.NewLabel("Translate Prompt"),
		f.translatePrompt,
		widget.NewLabelWithStyle(tpHint, fyne.TextAlignLeading, fyne.TextStyle{Italic: true}),
		widget.NewLabel("Screenshot Hotkey"),
		f.screenHK,
		widget.NewLabel("Clipboard OCR Hotkey"),
		f.clipHK,
		widget.NewLabel("Translate Hotkey"),
		f.translateHK,
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
	f.translateModel.SetText(def.TranslateModel)
	f.translatePrompt.SetText(def.TranslatePrompt)
	f.screenHK.SetText(def.ScreenshotHotkey)
	f.clipHK.SetText(def.ClipboardOCRHotkey)
	f.translateHK.SetText(def.TranslateHotkey)
}

func trySave(win fyne.Window, f *settingsFields, deps SettingsWindowDeps) {
	newCfg := &Config{
		OCREndpoint:        f.endpoint.Text,
		OCRModel:           f.model.Text,
		OCRPrompt:          f.prompt.Text,
		TranslateModel:     f.translateModel.Text,
		TranslatePrompt:    f.translatePrompt.Text,
		ScreenshotHotkey:   f.screenHK.Text,
		ClipboardOCRHotkey: f.clipHK.Text,
		TranslateHotkey:    f.translateHK.Text,
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
	if _, err := ParseHotkey(cfg.TranslateHotkey); err != nil {
		return err
	}
	return nil
}
