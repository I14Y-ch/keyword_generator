#!/bin/bash
# Setup script for the Keyword Generator application

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Setup complete!"
echo "Note: spaCy models are optional and will be loaded at runtime if available."
echo "To download models locally for testing, run:"
echo "  python -m spacy download de_core_news_md"
