package main

import (
	"fmt"
	"strings"

	"golang.design/x/hotkey"
)

type HotkeyBinding struct {
	Mods []hotkey.Modifier
	Key  hotkey.Key
}

var modifierMap = map[string]hotkey.Modifier{
	"cmd":     hotkey.ModCmd,
	"command": hotkey.ModCmd,
	"shift":   hotkey.ModShift,
	"ctrl":    hotkey.ModCtrl,
	"control": hotkey.ModCtrl,
	"alt":     hotkey.ModOption,
	"option":  hotkey.ModOption,
	"opt":     hotkey.ModOption,
}

var keyMap = buildKeyMap()

func buildKeyMap() map[string]hotkey.Key {
	m := map[string]hotkey.Key{
		"space":  hotkey.KeySpace,
		"return": hotkey.KeyReturn,
		"enter":  hotkey.KeyReturn,
		"escape": hotkey.KeyEscape,
		"esc":    hotkey.KeyEscape,
		"tab":    hotkey.KeyTab,
		"delete": hotkey.KeyDelete,
		"left":   hotkey.KeyLeft,
		"right":  hotkey.KeyRight,
		"up":     hotkey.KeyUp,
		"down":   hotkey.KeyDown,
	}
	letters := []hotkey.Key{
		hotkey.KeyA, hotkey.KeyB, hotkey.KeyC, hotkey.KeyD, hotkey.KeyE,
		hotkey.KeyF, hotkey.KeyG, hotkey.KeyH, hotkey.KeyI, hotkey.KeyJ,
		hotkey.KeyK, hotkey.KeyL, hotkey.KeyM, hotkey.KeyN, hotkey.KeyO,
		hotkey.KeyP, hotkey.KeyQ, hotkey.KeyR, hotkey.KeyS, hotkey.KeyT,
		hotkey.KeyU, hotkey.KeyV, hotkey.KeyW, hotkey.KeyX, hotkey.KeyY,
		hotkey.KeyZ,
	}
	for i, k := range letters {
		m[string(rune('a'+i))] = k
	}
	digits := []hotkey.Key{
		hotkey.Key0, hotkey.Key1, hotkey.Key2, hotkey.Key3, hotkey.Key4,
		hotkey.Key5, hotkey.Key6, hotkey.Key7, hotkey.Key8, hotkey.Key9,
	}
	for i, k := range digits {
		m[string(rune('0'+i))] = k
	}
	fkeys := []hotkey.Key{
		hotkey.KeyF1, hotkey.KeyF2, hotkey.KeyF3, hotkey.KeyF4, hotkey.KeyF5,
		hotkey.KeyF6, hotkey.KeyF7, hotkey.KeyF8, hotkey.KeyF9, hotkey.KeyF10,
		hotkey.KeyF11, hotkey.KeyF12,
	}
	for i, k := range fkeys {
		m[fmt.Sprintf("f%d", i+1)] = k
	}
	return m
}

func ParseHotkey(s string) (HotkeyBinding, error) {
	parts := strings.Split(strings.ToLower(strings.TrimSpace(s)), "+")
	if len(parts) < 2 {
		return HotkeyBinding{}, fmt.Errorf("hotkey needs modifier(s) and a key: %q", s)
	}
	var mods []hotkey.Modifier
	for _, p := range parts[:len(parts)-1] {
		m, ok := modifierMap[strings.TrimSpace(p)]
		if !ok {
			return HotkeyBinding{}, fmt.Errorf("unknown modifier: %q", p)
		}
		mods = append(mods, m)
	}
	last := strings.TrimSpace(parts[len(parts)-1])
	k, ok := keyMap[last]
	if !ok {
		return HotkeyBinding{}, fmt.Errorf("unknown key: %q", last)
	}
	return HotkeyBinding{Mods: mods, Key: k}, nil
}

type HotkeyManager struct {
	screenshot   *hotkey.Hotkey
	clipboard    *hotkey.Hotkey
	translate    *hotkey.Hotkey
	onScreenshot func()
	onClipboard  func()
	onTranslate  func()
}

type HotkeyCallbacks struct {
	OnScreenshot func()
	OnClipboard  func()
	OnTranslate  func()
}

func NewHotkeyManager(cb HotkeyCallbacks) *HotkeyManager {
	return &HotkeyManager{
		onScreenshot: cb.OnScreenshot,
		onClipboard:  cb.OnClipboard,
		onTranslate:  cb.OnTranslate,
	}
}

func (m *HotkeyManager) RegisterAll(cfg *Config) error {
	m.Unregister()

	screenshot, err := ParseHotkey(cfg.ScreenshotHotkey)
	if err != nil {
		return fmt.Errorf("screenshot hotkey: %w", err)
	}
	clip, err := ParseHotkey(cfg.ClipboardOCRHotkey)
	if err != nil {
		return fmt.Errorf("clipboard hotkey: %w", err)
	}
	translate, err := ParseHotkey(cfg.TranslateHotkey)
	if err != nil {
		return fmt.Errorf("translate hotkey: %w", err)
	}

	m.screenshot, err = bindHotkey(screenshot, m.onScreenshot)
	if err != nil {
		return fmt.Errorf("register screenshot hotkey: %w", err)
	}
	m.clipboard, err = bindHotkey(clip, m.onClipboard)
	if err != nil {
		return fmt.Errorf("register clipboard hotkey: %w", err)
	}
	m.translate, err = bindHotkey(translate, m.onTranslate)
	if err != nil {
		return fmt.Errorf("register translate hotkey: %w", err)
	}
	return nil
}

func bindHotkey(b HotkeyBinding, callback func()) (*hotkey.Hotkey, error) {
	hk := hotkey.New(b.Mods, b.Key)
	if err := hk.Register(); err != nil {
		return nil, err
	}
	go listenHotkey(hk, callback)
	return hk, nil
}

func listenHotkey(hk *hotkey.Hotkey, callback func()) {
	for range hk.Keydown() {
		go callback()
	}
}

func (m *HotkeyManager) Unregister() {
	if m.screenshot != nil {
		m.screenshot.Unregister()
		m.screenshot = nil
	}
	if m.clipboard != nil {
		m.clipboard.Unregister()
		m.clipboard = nil
	}
	if m.translate != nil {
		m.translate.Unregister()
		m.translate = nil
	}
}

func (m *HotkeyManager) Reload(cfg *Config) error {
	return m.RegisterAll(cfg)
}
