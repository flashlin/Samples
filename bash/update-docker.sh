# Update the package index
sudo apt update

# Upgrade Docker
sudo apt upgrade docker-ce

# Check the current version of Docker Compose
docker-compose version

# Download the latest version of Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Apply executable permissions to the binary
sudo chmod +x /usr/local/bin/docker-compose

# Verify the installation
docker-compose version