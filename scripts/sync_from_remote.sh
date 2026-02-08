#!/bin/bash
# Sync files from remote SSH server back to local

# Configuration
LOCAL_PATH="$(dirname $(dirname $(realpath $0)))"
CONFIG_FILE="$LOCAL_PATH/.sync_config"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Load configuration
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
else
    echo -e "${RED}Error: Configuration file not found.${NC}"
    echo "Please run: ./sync_to_remote.sh --setup"
    exit 1
fi

# Function to show usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -c          Sync only checkpoints and model files"
    echo "  -l          Sync only logs and results"
    echo "  -d          Dry run - show what would be synced"
    echo "  --help      Show this help message"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoints)
            CHECKPOINTS_ONLY=1
            shift
            ;;
        -l|--logs)
            LOGS_ONLY=1
            shift
            ;;
        -d|--dry-run)
            DRY_RUN="--dry-run"
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

# Sync specific directories
if [ "$CHECKPOINTS_ONLY" = "1" ]; then
    echo -e "${YELLOW}Syncing checkpoints from remote...${NC}"
    rsync -avz --progress $DRY_RUN \
        "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/checkpoints/" \
        "$LOCAL_PATH/checkpoints/"
elif [ "$LOGS_ONLY" = "1" ]; then
    echo -e "${YELLOW}Syncing logs from remote...${NC}"
    rsync -avz --progress $DRY_RUN \
        "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/wandb/" \
        "$LOCAL_PATH/wandb/"
    rsync -avz --progress $DRY_RUN \
        "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/results/" \
        "$LOCAL_PATH/results/"
else
    # Full sync from remote
    echo -e "${YELLOW}Full sync from remote...${NC}"
    echo -e "${GREEN}Syncing from: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH${NC}"
    echo -e "${GREEN}Syncing to:   $LOCAL_PATH${NC}"
    
    rsync -avz --progress \
        --exclude=".git/" \
        --exclude="__pycache__/" \
        --exclude="*.pyc" \
        $DRY_RUN \
        "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/" \
        "$LOCAL_PATH/"
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Sync completed successfully!${NC}"
else
    echo -e "${RED}Sync failed!${NC}"
    exit 1
fi