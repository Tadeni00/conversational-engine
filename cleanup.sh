# 1. Clean apt cache
sudo apt-get clean

# 2. Remove unused packages and dependencies
sudo apt-get autoremove -y

# 3. Remove npm cache (if you're using Node.js)
npm cache clean --force

# 4. Remove pip cache (if using Python)
rm -rf ~/.cache/pip

# 5. Remove Docker images (if you're using Docker)
docker system prune -a --volumes -f

# 6. Remove unused Python environments (if using venv or virtualenv)
find ~ -type d -name "__pycache__" -exec rm -rf {} +

# 7. Remove log files and other cruft
rm -rf ~/.local/share/Trash/*
rm -rf ~/Library/Logs/*
rm -rf ~/logs/*

# 8. Check disk usage
df -h
