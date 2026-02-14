package main

import (
	"strings"
	"unicode"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
)

func (a *AppState) buildListView() {
	a.Table = tview.NewTable().
		SetSelectable(true, false).
		SetFixed(1, 0).
		SetSelectedStyle(tcell.StyleDefault.Background(tcell.ColorDarkBlue).Foreground(tcell.ColorWhite))

	a.Table.SetBorder(false)

	a.InputField = tview.NewInputField().
		SetFieldBackgroundColor(tcell.ColorDefault)

	a.StatusBar = tview.NewTextView().
		SetTextColor(tcell.ColorGray)
	a.clearStatus()

	a.ListFlex = tview.NewFlex().SetDirection(tview.FlexRow).
		AddItem(a.Table, 0, 1, true).
		AddItem(a.StatusBar, 1, 0, false)

	a.setupTableInput()
	a.setupInputFieldHandler()
}

func (a *AppState) setupTableInput() {
	a.Table.SetInputCapture(func(event *tcell.EventKey) *tcell.EventKey {
		switch event.Rune() {
		case '/':
			a.enterFilterMode()
			return nil
		case ':':
			a.enterCommandMode()
			return nil
		case 'q':
			a.App.Stop()
			return nil
		}

		switch event.Key() {
		case tcell.KeyEscape:
			a.resetFilter()
			return nil
		}

		return event
	})

	a.Table.SetSelectionChangedFunc(func(row, column int) {
		if row > 0 {
			a.SelectedRow = row - 1
		}
	})
}

func (a *AppState) enterFilterMode() {
	a.Mode = ModeFilter
	a.showInput("/")
}

func (a *AppState) enterCommandMode() {
	a.Mode = ModeCommand
	a.showInput(":")
}

func (a *AppState) resetFilter() {
	a.FilterText = ""
	a.applyFilter()
	a.renderTable()
	a.clearStatus()
}

func (a *AppState) setupInputFieldHandler() {
	a.InputField.SetChangedFunc(func(text string) {
		if a.Mode == ModeFilter {
			a.FilterText = text
			a.applyFilter()
			a.renderTable()
		}
	})

	a.InputField.SetDoneFunc(func(key tcell.Key) {
		switch key {
		case tcell.KeyEnter:
			a.handleInputEnter()
		case tcell.KeyEscape:
			a.handleInputEscape()
		}
	})
}

func (a *AppState) handleInputEnter() {
	text := a.InputField.GetText()
	mode := a.Mode
	a.hideInput()

	switch mode {
	case ModeFilter:
		// keep filter applied
	case ModeCommand:
		a.executeCommand(strings.TrimSpace(text))
	}
}

func (a *AppState) handleInputEscape() {
	if a.Mode == ModeFilter {
		a.FilterText = ""
		a.applyFilter()
		a.renderTable()
	}
	a.hideInput()
}

func matchesFuzzy(target, pattern string) bool {
	if pattern == "" {
		return true
	}

	lowerTarget := strings.ToLower(target)
	lowerPattern := strings.ToLower(pattern)

	if strings.Contains(lowerTarget, lowerPattern) {
		return true
	}

	return matchesSequential(lowerTarget, lowerPattern)
}

func matchesSequential(target, pattern string) bool {
	targetRunes := []rune(target)
	patternRunes := []rune(pattern)

	ti := 0
	for _, p := range patternRunes {
		found := false
		for ti < len(targetRunes) {
			if equalFold(targetRunes[ti], p) {
				ti++
				found = true
				break
			}
			ti++
		}
		if !found {
			return false
		}
	}
	return true
}

func equalFold(a, b rune) bool {
	return unicode.ToLower(a) == unicode.ToLower(b)
}
