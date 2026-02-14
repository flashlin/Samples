package main

import (
	"bufio"
	"context"
	"fmt"
	"strings"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
)

type InputMode int

const (
	ModeNormal InputMode = iota
	ModeFilter
	ModeCommand
)

type AppState struct {
	App         *tview.Application
	Pages       *tview.Pages
	Docker      *DockerClient
	Containers  []ContainerInfo
	FilteredIdx []int
	SelectedRow int
	Mode        InputMode
	FilterText  string
	Highlighted map[int]bool

	Table      *tview.Table
	ListFlex   *tview.Flex
	InputField *tview.InputField
	StatusBar  *tview.TextView

	LogView   *tview.TextView
	LogFlex   *tview.Flex
	LogInput  *tview.InputField
	LogCancel context.CancelFunc
	LogSearch string
}

func NewApp(docker *DockerClient) *AppState {
	app := &AppState{
		App:         tview.NewApplication(),
		Pages:       tview.NewPages(),
		Docker:      docker,
		Mode:        ModeNormal,
		Highlighted: make(map[int]bool),
	}

	app.buildListView()
	app.buildLogView()

	app.Pages.AddPage("list", app.ListFlex, true, true)
	app.Pages.AddPage("log", app.LogFlex, true, false)

	return app
}

func (a *AppState) Run() error {
	if err := a.refreshContainers(); err != nil {
		return err
	}

	a.App.SetRoot(a.Pages, true)
	a.App.EnableMouse(false)
	return a.App.Run()
}

func (a *AppState) refreshContainers() error {
	containers, err := a.Docker.ListContainers()
	if err != nil {
		return err
	}

	a.Containers = containers
	a.applyFilter()
	a.renderTable()
	return nil
}

func (a *AppState) applyFilter() {
	a.FilteredIdx = a.FilteredIdx[:0]

	if a.FilterText == "" {
		for i := range a.Containers {
			a.FilteredIdx = append(a.FilteredIdx, i)
		}
		return
	}

	for i, c := range a.Containers {
		if matchesFuzzy(c.Name, a.FilterText) || matchesFuzzy(c.ID, a.FilterText) {
			a.FilteredIdx = append(a.FilteredIdx, i)
		}
	}
}

func (a *AppState) selectedContainer() *ContainerInfo {
	if a.SelectedRow < 0 || a.SelectedRow >= len(a.FilteredIdx) {
		return nil
	}
	idx := a.FilteredIdx[a.SelectedRow]
	return &a.Containers[idx]
}

func (a *AppState) toggleHighlight() {
	if a.SelectedRow < 0 || a.SelectedRow >= len(a.FilteredIdx) {
		return
	}
	idx := a.FilteredIdx[a.SelectedRow]
	if a.Highlighted[idx] {
		delete(a.Highlighted, idx)
	} else {
		a.Highlighted[idx] = true
	}
	a.renderTable()
}

func (a *AppState) clearHighlights() {
	a.Highlighted = make(map[int]bool)
	a.renderTable()
}

func (a *AppState) highlightedContainers() []*ContainerInfo {
	if len(a.Highlighted) == 0 {
		c := a.selectedContainer()
		if c == nil {
			return nil
		}
		return []*ContainerInfo{c}
	}

	var result []*ContainerInfo
	for idx := range a.Highlighted {
		if idx >= 0 && idx < len(a.Containers) {
			result = append(result, &a.Containers[idx])
		}
	}
	return result
}

func (a *AppState) showInput(prefix string) {
	a.InputField.SetLabel(prefix)
	a.InputField.SetText("")
	a.ListFlex.AddItem(a.InputField, 1, 0, true)
	a.App.SetFocus(a.InputField)
}

func (a *AppState) hideInput() {
	a.ListFlex.RemoveItem(a.InputField)
	a.App.SetFocus(a.Table)
	a.Mode = ModeNormal
}

func (a *AppState) setStatus(msg string) {
	a.StatusBar.SetText(" " + msg)
}

func (a *AppState) clearStatus() {
	a.StatusBar.SetText(" q:quit  /:filter  ::command  SPACE:select")
}

func (a *AppState) switchToLog(c *ContainerInfo) {
	a.LogView.Clear()
	a.LogSearch = ""
	a.Pages.SwitchToPage("log")

	ctx, cancel := context.WithCancel(context.Background())
	a.LogCancel = cancel

	reader, err := a.Docker.StreamLogs(ctx, c.ID)
	if err != nil {
		a.setStatus("Log error: " + err.Error())
		return
	}

	go func() {
		scanner := bufio.NewScanner(reader)
		scanner.Buffer(make([]byte, 64*1024), 1024*1024)
		for scanner.Scan() {
			line := scanner.Text()
			a.App.QueueUpdateDraw(func() {
				fmt.Fprintln(a.LogView, line)
			})
		}
		reader.Close()
	}()
}

func (a *AppState) switchToList() {
	if a.LogCancel != nil {
		a.LogCancel()
		a.LogCancel = nil
	}
	a.hideLogInput()
	a.Pages.SwitchToPage("list")
	a.App.SetFocus(a.Table)
}

func (a *AppState) executeCommand(cmd string) {
	if cmd == "q" {
		a.App.Stop()
		return
	}

	targets := a.highlightedContainers()
	if len(targets) == 0 {
		a.setStatus("No container selected")
		return
	}

	switch cmd {
	case "log":
		a.switchToLog(targets[0])
	case "stop":
		a.executeBatchAction(targets, "Stopping", a.Docker.StopContainer)
	case "start":
		a.executeBatchAction(targets, "Starting", a.Docker.StartContainer)
	case "restart":
		a.executeBatchAction(targets, "Restarting", a.Docker.RestartContainer)
	case "rm":
		a.executeBatchAction(targets, "Removing", a.Docker.RemoveContainer)
	case "bash":
		a.execBash(targets[0])
	default:
		a.setStatus("Unknown command: " + cmd)
	}
}

func (a *AppState) executeBatchAction(targets []*ContainerInfo, action string, fn func(string) error) {
	names := a.formatContainerNames(targets)
	a.setStatus(action + " " + names + "...")

	go func() {
		var errors []string
		for _, c := range targets {
			if err := fn(c.ID); err != nil {
				errors = append(errors, c.Name+": "+err.Error())
			}
		}

		a.App.QueueUpdateDraw(func() {
			if len(errors) > 0 {
				a.setStatus("Error: " + strings.Join(errors, "; "))
				return
			}
			a.clearHighlights()
			_ = a.refreshContainers()
			a.clearStatus()
		})
	}()
}

func (a *AppState) formatContainerNames(targets []*ContainerInfo) string {
	if len(targets) == 1 {
		return targets[0].Name
	}
	names := make([]string, len(targets))
	for i, c := range targets {
		names[i] = c.Name
	}
	return fmt.Sprintf("%d containers", len(names))
}

func (a *AppState) execBash(c *ContainerInfo) {
	a.App.Suspend(func() {
		err := a.Docker.ExecBash(c.ID)
		if err != nil {
			a.setStatus("Bash error: " + err.Error())
		}
	})
}

func (a *AppState) renderTable() {
	a.Table.Clear()

	headers := []string{"ID", "NAME", "PORTS", "STATUS"}
	for col, h := range headers {
		cell := tview.NewTableCell(h).
			SetTextColor(tcell.ColorYellow).
			SetSelectable(false).
			SetExpansion(expansionForColumn(col))
		a.Table.SetCell(0, col, cell)
	}

	for row, idx := range a.FilteredIdx {
		c := a.Containers[idx]
		color := tcell.ColorRed
		if c.IsRunning() {
			color = tcell.ColorGreen
		}

		highlighted := a.Highlighted[idx]

		cells := []string{c.ShortID(), truncateText(c.Name, 50), c.Ports, c.Status}
		for col, text := range cells {
			cell := tview.NewTableCell(text).
				SetTextColor(color).
				SetMaxWidth(maxWidthForColumn(col)).
				SetExpansion(expansionForColumn(col))
			if highlighted {
				cell.SetBackgroundColor(tcell.ColorDarkGoldenrod)
			}
			a.Table.SetCell(row+1, col, cell)
		}
	}

	if a.SelectedRow >= len(a.FilteredIdx) {
		a.SelectedRow = len(a.FilteredIdx) - 1
	}
	if a.SelectedRow < 0 {
		a.SelectedRow = 0
	}
	if len(a.FilteredIdx) > 0 {
		a.Table.Select(a.SelectedRow+1, 0)
	}
}

func expansionForColumn(col int) int {
	switch col {
	case 0:
		return 1
	case 1:
		return 2
	case 2:
		return 2
	case 3:
		return 2
	default:
		return 1
	}
}

func maxWidthForColumn(col int) int {
	switch col {
	case 1:
		return 50
	default:
		return 0
	}
}

func truncateText(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
