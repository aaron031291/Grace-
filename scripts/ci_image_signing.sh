#!/bin/bash
# ci_image_signing.sh: Sign Docker images with cosign in CI/CD
set -e

IMAGE_NAME="grace-api:latest"
COSIGN_KEY="${COSIGN_KEY:-cosign.key}"

# Generate key if not present
if [ ! -f "$COSIGN_KEY" ]; then
  cosign generate-key-pair
fi

# Sign image
cosign sign --key "$COSIGN_KEY" "$IMAGE_NAME"

# Verify signature (for promotion/deploy)
cosign verify --key "$COSIGN_KEY.pub" "$IMAGE_NAME"

echo "Image $IMAGE_NAME signed and verified with cosign."
