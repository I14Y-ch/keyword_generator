#!/bin/bash
# Deploy script for Digital Ocean continuous deployment
# This script is called automatically on git push

# Exit on error
set -e

echo "Starting deployment..."

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy models
python setup.py

# Run any pending migrations or setup tasks (if needed)
# python manage.py migrate

echo "✓ Deployment completed successfully!"
