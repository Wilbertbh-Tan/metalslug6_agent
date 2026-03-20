#!/bin/bash
# Docker command aliases for Metal Slug 6 agent
# Source this file: source docker-aliases.sh

# Container management
alias mslug-stop='docker stop mslug6 && docker rm mslug6'
alias mslug-build='docker build -t metalslug-rl .'
alias mslug-run='docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --restart unless-stopped -d -p 5900:5900 -p 6080:6080 -v ~/Downloads:/games -v ~/Documents/Github/metalslug6_agent/outputs:/app/outputs -e RETROARCH_CORE=flycast_libretro -e RETROARCH_CONTENT=/games/mslug6.zip -e MODE=game-only --name mslug6 metalslug-rl'

# Log viewing
alias mslug-logs='docker logs mslug6'
alias mslug-logs-follow='docker logs -f mslug6'
alias mslug-logs-tail='docker logs --tail 50 mslug6'
alias mslug-logs-boot='docker logs mslug6 2>&1 | grep -i "auto-boot\|sent:\|gameplay\|save state"'
alias mslug-logs-keys='docker logs mslug6 2>&1 | grep -E "sent:|key sequence|BOOT_KEYS"'

# Debug logs

# Container status
alias mslug-ps='docker ps -a --filter name=mslug6'
alias mslug-status='docker ps -a --filter name=mslug6 --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'

# Quick restart (stop, rebuild, run)
alias mslug-restart='docker stop mslug6 2>/dev/null; docker rm mslug6 2>/dev/null; docker build -t metalslug-rl . && docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --restart unless-stopped -d -p 5900:5900 -p 6080:6080 -v ~/Downloads:/games -v ~/Documents/Github/metalslug6_agent/outputs:/app/outputs -e RETROARCH_CORE=flycast_libretro -e RETROARCH_CONTENT=/games/mslug6.zip -e MODE=game-only --name mslug6 metalslug-rl'

echo "Metal Slug Docker aliases loaded!"
echo "Available commands:"
echo "  mslug-stop       - Stop and remove container"
echo "  mslug-build      - Rebuild Docker image"
echo "  mslug-run        - Start container"
echo "  mslug-restart    - Stop, rebuild, and start"
echo "  mslug-logs       - View container logs"
echo "  mslug-logs-follow - Follow logs in real-time"
echo "  mslug-logs-tail  - View last 50 lines"
echo "  mslug-logs-boot  - Filter for auto-boot messages"
echo "  mslug-logs-keys  - Filter for key sequence"
echo "  mslug-ps         - Show container status"
echo "  mslug-status     - Show container status (formatted)"


