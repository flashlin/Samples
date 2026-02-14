package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sort"
	"strings"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/client"
	"github.com/docker/docker/pkg/stdcopy"
)

type DockerClient struct {
	cli *client.Client
}

func NewDockerClient() (*DockerClient, error) {
	cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return nil, fmt.Errorf("failed to create docker client: %w", err)
	}
	return &DockerClient{cli: cli}, nil
}

func (d *DockerClient) Close() error {
	return d.cli.Close()
}

func (d *DockerClient) ListContainers() ([]ContainerInfo, error) {
	containers, err := d.cli.ContainerList(context.Background(), container.ListOptions{All: true})
	if err != nil {
		return nil, fmt.Errorf("failed to list containers: %w", err)
	}

	result := make([]ContainerInfo, 0, len(containers))
	for _, c := range containers {
		result = append(result, convertToContainerInfo(c))
	}

	sort.Slice(result, func(i, j int) bool {
		return result[i].Name < result[j].Name
	})

	return result, nil
}

func convertToContainerInfo(c types.Container) ContainerInfo {
	name := ""
	if len(c.Names) > 0 {
		name = strings.TrimPrefix(c.Names[0], "/")
	}

	return ContainerInfo{
		ID:      c.ID,
		Name:    name,
		Status:  c.Status,
		State:   c.State,
		Ports:   formatPorts(c.Ports),
		Created: time.Unix(c.Created, 0),
	}
}

func formatPorts(ports []types.Port) string {
	if len(ports) == 0 {
		return ""
	}

	seen := make(map[string]bool)
	var parts []string

	for _, p := range ports {
		var entry string
		if p.PublicPort != 0 {
			entry = fmt.Sprintf("%d->%d/%s", p.PublicPort, p.PrivatePort, p.Type)
		} else {
			entry = fmt.Sprintf("%d/%s", p.PrivatePort, p.Type)
		}
		if !seen[entry] {
			seen[entry] = true
			parts = append(parts, entry)
		}
	}

	return strings.Join(parts, ", ")
}

func (d *DockerClient) StopContainer(id string) error {
	return d.cli.ContainerStop(context.Background(), id, container.StopOptions{})
}

func (d *DockerClient) StartContainer(id string) error {
	return d.cli.ContainerStart(context.Background(), id, container.StartOptions{})
}

func (d *DockerClient) RestartContainer(id string) error {
	return d.cli.ContainerRestart(context.Background(), id, container.StopOptions{})
}

func (d *DockerClient) RemoveContainer(id string) error {
	return d.cli.ContainerRemove(context.Background(), id, container.RemoveOptions{Force: true})
}

func (d *DockerClient) StreamLogs(ctx context.Context, id string) (io.ReadCloser, error) {
	reader, err := d.cli.ContainerLogs(ctx, id, container.LogsOptions{
		ShowStdout: true,
		ShowStderr: true,
		Follow:     true,
		Tail:       "100",
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get container logs: %w", err)
	}

	pr, pw := io.Pipe()
	go func() {
		defer pw.Close()
		_, _ = stdcopy.StdCopy(pw, pw, reader)
	}()

	go func() {
		<-ctx.Done()
		reader.Close()
	}()

	return pr, nil
}

func (d *DockerClient) ExecBash(containerID string) error {
	cmd := exec.Command("docker", "exec", "-it", containerID, "/bin/bash")
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
