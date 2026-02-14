package main

import (
	"fmt"
	"time"
)

type ContainerInfo struct {
	ID      string
	Name    string
	Status  string
	State   string
	Ports   string
	Created time.Time
}

func (c ContainerInfo) ShortID() string {
	if len(c.ID) > 12 {
		return c.ID[:12]
	}
	return c.ID
}

func (c ContainerInfo) IsRunning() bool {
	return c.State == "running"
}

func (c ContainerInfo) FormatUptime() string {
	elapsed := time.Since(c.Created)

	switch {
	case elapsed < time.Minute:
		return fmt.Sprintf("%ds", int(elapsed.Seconds()))
	case elapsed < time.Hour:
		return fmt.Sprintf("%dm", int(elapsed.Minutes()))
	case elapsed < 24*time.Hour:
		return fmt.Sprintf("%dh", int(elapsed.Hours()))
	default:
		return fmt.Sprintf("%dd", int(elapsed.Hours()/24))
	}
}
