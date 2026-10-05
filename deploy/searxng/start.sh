#!/usr/bin/env bash
# Start SearXNG development container for AURA (M14)
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CONTAINER_NAME="aura-searxng"
PORT="${SEARXNG_PORT:-8080}"

if podman ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Starting existing container ${CONTAINER_NAME}..."
    podman start "${CONTAINER_NAME}"
else
    echo "Creating and starting ${CONTAINER_NAME} on port ${PORT}..."
    podman run -d \
        --name "${CONTAINER_NAME}" \
        -p "${PORT}:8080" \
        -v "${DIR}/settings.yml:/etc/searxng/settings.yml:Z" \
        docker.io/searxng/searxng:latest
fi

echo "SearXNG is running at http://localhost:${PORT}"
