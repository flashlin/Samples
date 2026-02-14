package main

import (
	"fmt"
	"strings"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
)

func (a *AppState) buildLogView() {
	a.LogView = tview.NewTextView().
		SetDynamicColors(true).
		SetScrollable(true)

	a.LogView.SetBorder(false)

	a.LogInput = tview.NewInputField().
		SetLabel("/").
		SetFieldBackgroundColor(tcell.ColorDefault)

	logStatus := tview.NewTextView().
		SetTextColor(tcell.ColorGray).
		SetText(" q:back  /:search  G:bottom")

	a.LogFlex = tview.NewFlex().SetDirection(tview.FlexRow).
		AddItem(a.LogView, 0, 1, true).
		AddItem(logStatus, 1, 0, false)

	a.setupLogViewInput()
	a.setupLogInputHandler()
}

func (a *AppState) setupLogViewInput() {
	a.LogView.SetInputCapture(func(event *tcell.EventKey) *tcell.EventKey {
		switch event.Rune() {
		case 'q':
			a.switchToList()
			return nil
		case '/':
			a.showLogInput()
			return nil
		case 'G':
			a.LogView.ScrollToEnd()
			return nil
		case 'g':
			a.LogView.ScrollToBeginning()
			return nil
		}

		switch event.Key() {
		case tcell.KeyEscape:
			a.switchToList()
			return nil
		}

		return event
	})
}

func (a *AppState) showLogInput() {
	a.LogInput.SetText("")
	a.LogFlex.AddItem(a.LogInput, 1, 0, true)
	a.App.SetFocus(a.LogInput)
}

func (a *AppState) hideLogInput() {
	a.LogFlex.RemoveItem(a.LogInput)
	a.App.SetFocus(a.LogView)
}

func (a *AppState) setupLogInputHandler() {
	a.LogInput.SetDoneFunc(func(key tcell.Key) {
		switch key {
		case tcell.KeyEnter:
			a.LogSearch = a.LogInput.GetText()
			a.highlightLogSearch()
			a.hideLogInput()
		case tcell.KeyEscape:
			a.hideLogInput()
		}
	})
}

func (a *AppState) highlightLogSearch() {
	if a.LogSearch == "" {
		return
	}

	original := a.LogView.GetText(true)
	highlighted := highlightMatches(original, a.LogSearch)
	a.LogView.SetText(highlighted)
}

func highlightMatches(text, search string) string {
	if search == "" {
		return text
	}

	lowerText := strings.ToLower(text)
	lowerSearch := strings.ToLower(search)

	var builder strings.Builder
	idx := 0

	for idx < len(lowerText) {
		pos := strings.Index(lowerText[idx:], lowerSearch)
		if pos < 0 {
			builder.WriteString(text[idx:])
			break
		}

		builder.WriteString(text[idx : idx+pos])
		matchEnd := idx + pos + len(search)
		builder.WriteString(fmt.Sprintf("[yellow::b]%s[-:-:-]", text[idx+pos:matchEnd]))
		idx = matchEnd
	}

	return builder.String()
}
