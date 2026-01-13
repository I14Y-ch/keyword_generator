#!/bin/bash
# Deploy script for Digital Ocean continuous deployment
# This script is called automatically on git push

# Exit on error
set -e

echo "Starting deployment..."

# Ensure Hunspell dictionaries are available (needed for compound splitting)
if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" -eq 0 ]; then
	echo "Installing Hunspell dictionaries..."
	export DEBIAN_FRONTEND=noninteractive
	apt-get update -y
	apt-get install -y --no-install-recommends \
		hunspell \
		hunspell-de-ch \
		hunspell-fr \
		hunspell-it \
		hunspell-en-gb
else
	echo "Skipping Hunspell installation (apt-get not available or insufficient permissions)"
fi

# Install Python dependencies
pip install -r requirements.txt

echo "✓ Deployment completed successfully!"
