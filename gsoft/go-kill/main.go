package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"syscall"

	"github.com/shirou/gopsutil/v3/process"
)

type ProcessTarget struct {
	PID  int32
	Name string
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go-kill <script.sh>")
		os.Exit(1)
	}

	targetScript := os.Args[1]

	targetProcesses := findTargetProcesses(targetScript)

	if len(targetProcesses) == 0 {
		fmt.Printf("No processes found for %s\n", targetScript)
		os.Exit(0)
	}

	displayTargetProcesses(targetProcesses)
	forceKill := askUserKillStrategy()
	terminateProcesses(targetProcesses, forceKill)
}

func findTargetProcesses(targetScript string) []ProcessTarget {
	allProcesses, err := process.Processes()
	if err != nil {
		fmt.Printf("Error getting processes: %v\n", err)
		return nil
	}

	var targets []ProcessTarget
	rootPIDs := findRootProcesses(allProcesses, targetScript)

	for _, pid := range rootPIDs {
		childProcesses := findChildProcessesRecursive(pid)
		for _, child := range childProcesses {
			targets = append(targets, child)
		}

		rootProc, err := process.NewProcess(pid)
		if err == nil {
			name, _ := rootProc.Name()
			targets = append(targets, ProcessTarget{PID: pid, Name: name})
		}
	}

	return deduplicateTargets(targets)
}

func findRootProcesses(allProcesses []*process.Process, targetScript string) []int32 {
	var pids []int32
	for _, p := range allProcesses {
		cmdline, err := p.Cmdline()
		if err == nil && strings.Contains(cmdline, targetScript) && !strings.Contains(cmdline, "go-kill") {
			pids = append(pids, p.Pid)
		}
	}
	return pids
}

func findChildProcessesRecursive(parentPID int32) []ProcessTarget {
	var targets []ProcessTarget
	p, err := process.NewProcess(parentPID)
	if err != nil {
		return targets
	}

	children, err := p.Children()
	if err != nil || len(children) == 0 {
		return targets
	}

	for _, child := range children {
		childTargets := findChildProcessesRecursive(child.Pid)
		targets = append(targets, childTargets...)

		name, _ := child.Name()
		targets = append(targets, ProcessTarget{PID: child.Pid, Name: name})
	}

	return targets
}

func deduplicateTargets(targets []ProcessTarget) []ProcessTarget {
	seen := make(map[int32]bool)
	var uniqueTargets []ProcessTarget
	for _, target := range targets {
		if !seen[target.PID] {
			seen[target.PID] = true
			uniqueTargets = append(uniqueTargets, target)
		}
	}
	return uniqueTargets
}

func displayTargetProcesses(targets []ProcessTarget) {
	fmt.Printf("Found %d processes to kill:\n", len(targets))
	for _, t := range targets {
		fmt.Printf("PID: %d (Name: %s)\n", t.PID, t.Name)
	}
}

func askUserKillStrategy() bool {
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("正常結束或是強制結束(K/f)? ")
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)
	input = strings.ToLower(input)

	return input == "f"
}

func terminateProcesses(targets []ProcessTarget, forceKill bool) {
	signal := syscall.SIGTERM
	if forceKill {
		signal = syscall.SIGKILL
	}

	for _, target := range targets {
		p, err := process.NewProcess(target.PID)
		if err != nil {
			continue
		}

		err = p.SendSignal(signal)
		if err != nil {
			fmt.Printf("Failed to kill PID %d: %v\n", target.PID, err)
		} else {
			fmt.Printf("Signal %v sent to PID %d (%s)\n", signal, target.PID, target.Name)
		}
	}
}
