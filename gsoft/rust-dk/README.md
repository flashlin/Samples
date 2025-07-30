# dk - Docker Container Management Tool

A friendly and interactive Docker container management tool written in Rust.

## Features

- 🐳 **Interactive Docker Management**: Friendly command-line interface for Docker containers
- 📊 **Container Status Overview**: View all running and stopped containers at a glance
- 📝 **Interactive Logs**: Browse and view container logs with filtering
- 🖥️ **Shell Access**: Enter container bash/shell environments easily
- 🗑️ **Safe Container Removal**: Remove containers with confirmation prompts
- 🔄 **Container Restart**: Restart containers interactively
- 🔍 **Real-time Filtering**: Filter containers by name or ID
- ⌨️ **Keyboard Navigation**: Full keyboard-driven interface

## Installation

### Prerequisites

- Rust 1.70+ 
- Docker installed and running
- Docker daemon accessible (usually requires Docker Desktop or Docker service running)

### Build from Source

```bash
git clone <repository-url>
cd dk
cargo build --release
```

The binary will be available at `target/release/dk`.

### Install Globally

```bash
cargo install --path .
```

## Usage

### Basic Commands

#### Show Container Status
```bash
dk
```
Displays all containers with their status, ports, and timing information.

#### View Container Logs
```bash
dk logs
```
Interactive selection of containers to view their logs.

#### Enter Container Shell
```bash
dk bash
```
Interactive selection of containers to enter their bash/shell environment.

#### Remove Containers
```bash
dk rm        # Safe removal with confirmation
dk rm -f     # Force removal with confirmation
```

#### Restart Containers
```bash
dk restart
```

### Interactive Features

#### Navigation
- **↑/↓ Arrow Keys**: Navigate through container list
- **Enter**: Select/confirm action
- **Space**: Toggle selection (in multi-select mode)
- **q/Esc**: Quit/cancel

#### Filtering
- **/**: Start filtering mode
- **Type**: Filter containers by name or ID in real-time
- **Esc**: Clear filter and exit filter mode

#### Multi-Selection
In commands like `rm`, `rm -f`, and `restart`:
- Use **Space** to toggle selection of multiple containers
- Selected containers are marked with ✓
- **Enter** to proceed with selected containers

## Examples

### View all containers
```bash
$ dk
🟢 Running Containers:
ID           NAME                           STATUS          PORTS                TIME
abc123456789 my-web-app                     Running         80:3000, 443:3001   2 hours ago
def987654321 redis-cache                    Running         6379:6379            1 day ago

🔴 Stopped Containers:
ID           NAME                           STATUS          PORTS                TIME
ghi456789123 old-database                   Exited (0)      -                    3 days ago
```

### Interactive log viewing
```bash
$ dk logs
# Opens interactive container selection
# Use arrow keys to navigate, Enter to view logs
# Use / to filter by container name or ID
```

### Safe container removal
```bash
$ dk rm
# Opens interactive selection with multi-select
# Use Space to select multiple containers
# Confirms before removal
```

## Configuration

dk uses Docker's default connection methods:
- Unix socket on Linux/macOS: `/var/run/docker.sock`
- Named pipe on Windows: `//./pipe/docker_engine`
- TCP connection if `DOCKER_HOST` environment variable is set

## Error Handling

dk provides friendly error messages for common issues:
- Docker daemon not running
- Permission issues
- Invalid container IDs
- Network connectivity problems

## Development

### Project Structure
```
src/
├── main.rs          # CLI entry point and argument parsing
├── commands.rs      # Command implementations
├── docker.rs        # Docker API client
├── types.rs         # Data structures and types
└── ui.rs           # Terminal UI and interactive components
```

### Dependencies
- **clap**: Command-line argument parsing
- **ratatui**: Terminal user interface
- **crossterm**: Cross-platform terminal manipulation
- **bollard**: Docker API client
- **tokio**: Async runtime
- **anyhow**: Error handling

### Building
```bash
cargo build          # Debug build
cargo build --release # Release build
cargo test           # Run tests
cargo clippy         # Linting
cargo fmt            # Code formatting
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run `cargo test` and `cargo clippy`
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Troubleshooting

### Docker daemon not accessible
```
Error: Failed to connect to Docker daemon. Make sure Docker is running.
```
**Solution**: Ensure Docker Desktop is running or Docker service is started.

### Permission denied
```
Error: Permission denied while trying to connect to Docker daemon
```
**Solution**: 
- On Linux: Add user to docker group: `sudo usermod -aG docker $USER`
- On macOS/Windows: Ensure Docker Desktop is running with proper permissions

### Container not found
```
Error: No such container: abc123
```
**Solution**: The container may have been removed. Run `dk` to see current containers.

## Roadmap

- [ ] Docker Compose support
- [ ] Container monitoring and stats
- [ ] Batch operations
- [ ] Configuration file support
- [ ] Custom themes and colors
- [ ] Export container information
- [ ] Integration with Docker registries