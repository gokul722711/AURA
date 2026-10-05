# SearXNG Local Deployment for AURA (M14)

SearXNG provides the open-source, free, local web-search backend for AURA's real web research pipeline.

## Prerequisites

- [Podman](https://podman.io/) (installed and running)

## Quick Start

Start the local SearXNG container:

```bash
./deploy/searxng/start.sh
```

Or manually with Podman:

```bash
podman run -d \
  --name aura-searxng \
  -p 8080:8080 \
  -v ./deploy/searxng/settings.yml:/etc/searxng/settings.yml:Z \
  docker.io/searxng/searxng:latest
```

## Verifying Local Service

Verify that the JSON search API is responding:

```bash
curl -s "http://localhost:8080/search?q=python&format=json" | jq .
```

## AURA Configuration

In your `.env`:

```env
AI_WEB_SEARCH_PROVIDER=searxng
AI_SEARXNG_URL=http://localhost:8080
```
