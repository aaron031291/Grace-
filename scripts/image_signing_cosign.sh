# Image Signing with Cosign
# Usage: cosign sign --key <key-path> grace_api:latest

# Example CI/CD step (after building and pushing image)
cosign sign --key $COSIGN_KEY_PATH $IMAGE_NAME

# To verify:
cosign verify --key $COSIGN_KEY_PATH $IMAGE_NAME

# For GitHub Actions, use the official cosign action:
# - uses: sigstore/cosign-installer@v3.2.0
# - run: cosign sign --key ${{ secrets.COSIGN_KEY }} $IMAGE_NAME
