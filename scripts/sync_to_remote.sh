#!/bin/bash
# Sync project files to remote SSH server

# Configuration - EDIT THESE VARIABLES
REMOTE_USER="timothyobiso"
REMOTE_HOST="aristotle.cs-i.brandeis.edu"
REMOTE_PATH="/home/timothyobiso/cais_course"
LOCAL_PATH="$(dirname $(dirname $(realpath $0)))"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if config file exists
CONFIG_FILE="$LOCAL_PATH/.sync_config"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    echo -e "${GREEN}Loaded configuration from .sync_config${NC}"
fi

# Function to show usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -u USER     Remote username (default: $REMOTE_USER)"
    echo "  -h HOST     Remote host (default: $REMOTE_HOST)"
    echo "  -p PATH     Remote path (default: $REMOTE_PATH)"
    echo "  -w          Watch mode - continuously sync changes"
    echo "  -d          Dry run - show what would be synced"
    echo "  --setup     Interactive setup to configure sync settings"
    echo "  --help      Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--user)
            REMOTE_USER="$2"
            shift 2
            ;;
        -h|--host)
            REMOTE_HOST="$2"
            shift 2
            ;;
        -p|--path)
            REMOTE_PATH="$2"
            shift 2
            ;;
        -w|--watch)
            WATCH_MODE=1
            shift
            ;;
        -d|--dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --setup)
            SETUP_MODE=1
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Interactive setup mode
if [ "$SETUP_MODE" = "1" ]; then
    echo -e "${YELLOW}Setting up sync configuration...${NC}"
    read -p "Remote username: " REMOTE_USER
    read -p "Remote host/IP: " REMOTE_HOST
    read -p "Remote project path: " REMOTE_PATH
    
    # Save configuration
    cat > "$CONFIG_FILE" << EOF
REMOTE_USER="$REMOTE_USER"
REMOTE_HOST="$REMOTE_HOST"
REMOTE_PATH="$REMOTE_PATH"
EOF
    
    echo -e "${GREEN}Configuration saved to .sync_config${NC}"
    
    # Test SSH connection
    echo -e "${YELLOW}Testing SSH connection...${NC}"
    ssh -o ConnectTimeout=5 "$REMOTE_USER@$REMOTE_HOST" "echo 'Connection successful!'"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}SSH connection successful!${NC}"
    else
        echo -e "${RED}SSH connection failed. Please check your settings.${NC}"
        exit 1
    fi
    
    # Create remote directory if it doesn't exist
    echo -e "${YELLOW}Creating remote directory...${NC}"
    ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_PATH"
    
    exit 0
fi

# Check if configuration is valid
if [ -z "$REMOTE_USER" ] || [ -z "$REMOTE_HOST" ] || [ -z "$REMOTE_PATH" ]; then
    echo -e "${RED}Error: Remote configuration not set.${NC}"
    echo "Please run: $0 --setup"
    exit 1
fi

# Rsync exclude patterns
EXCLUDE_FILE="$LOCAL_PATH/.rsyncignore"
if [ ! -f "$EXCLUDE_FILE" ]; then
    # Create default exclude file
    cat > "$EXCLUDE_FILE" << 'EOF'
.git/
.github/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/
.env
.venv/
venv/
ENV/
.DS_Store
*.log
.idea/
.vscode/
*.swp
*.swo
*~
.ipynb_checkpoints/
wandb/
checkpoints/
*.pt
*.pth
*.ckpt
EOF
    echo -e "${YELLOW}Created .rsyncignore file with default exclusions${NC}"
fi

# Main sync function
sync_files() {
    echo -e "${GREEN}Syncing from: $LOCAL_PATH${NC}"
    echo -e "${GREEN}Syncing to:   $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH${NC}"
    
    rsync -avz --progress \
        --exclude-from="$EXCLUDE_FILE" \
        --delete \
        -e "ssh -o PasswordAuthentication=yes -o PreferredAuthentications=password" \
        $DRY_RUN \
        "$LOCAL_PATH/" \
        "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Sync completed successfully!${NC}"
        return 0
    else
        echo -e "${RED}Sync failed!${NC}"
        return 1
    fi
}

# Watch mode using fswatch (macOS) or inotify (Linux)
watch_and_sync() {
    echo -e "${YELLOW}Starting watch mode. Press Ctrl+C to stop.${NC}"
    
    # Check which file watcher is available
    if command -v fswatch &> /dev/null; then
        # macOS with fswatch
        fswatch -o "$LOCAL_PATH" -e "\.git" -e "__pycache__" -e "\.pyc$" | while read f; do
            echo -e "${YELLOW}Changes detected, syncing...${NC}"
            sync_files
        done
    elif command -v inotifywait &> /dev/null; then
        # Linux with inotify-tools
        while true; do
            inotifywait -r -e modify,create,delete,move \
                --exclude '\.git|__pycache__|\.pyc$' \
                "$LOCAL_PATH"
            echo -e "${YELLOW}Changes detected, syncing...${NC}"
            sync_files
        done
    else
        echo -e "${RED}No file watcher found. Install fswatch (macOS) or inotify-tools (Linux)${NC}"
        echo "macOS: brew install fswatch"
        echo "Linux: sudo apt-get install inotify-tools"
        exit 1
    fi
}

# Main execution
if [ "$WATCH_MODE" = "1" ]; then
    sync_files
    watch_and_sync
else
    sync_files
fi
