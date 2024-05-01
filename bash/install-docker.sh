# Install Docker, you can ignore the warning from Docker about using WSL
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to the Docker group
sudo usermod -aG docker $USER

# Sanity check that both tools were installed successfully
docker --version
docker compose version

# Using Ubuntu 22.04 or Debian 10+? You need to do 1 extra step for iptables
# compatibility, you'll want to choose option (1) to use iptables-legacy from
# the prompt that'll come up when running the command below.
#
# You'll likely need to reboot Windows or at least restart WSL after applying
# this, otherwise networking inside of your containers won't work.
sudo update-alternatives --config iptables