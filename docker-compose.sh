#!/bin/bash
# ==============================================================================
# 使用方法：
#   ./docker-compose.sh up -d dl-dev-cuda11.8-cudnn8-pytorch1.13.1-ubuntu22.04
#   ./docker-compose.sh down
#   ./docker-compose.sh build dl-dev-cuda11.8-cudnn8-pytorch1.13.1-ubuntu22.04
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_DIR="$SCRIPT_DIR/my-dev-env/docker"

# 设置环境变量，供 docker-compose.yml 使用
export PROJECT_ROOT="$SCRIPT_DIR"
docker compose -f "$COMPOSE_DIR/docker-compose.yml" -f "$SCRIPT_DIR/docker-compose.yml" "$@"
