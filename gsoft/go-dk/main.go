package main

import (
	"fmt"
	"os"
)

func main() {
	docker, err := NewDockerClient()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	defer docker.Close()

	app := NewApp(docker)
	if err := app.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
