#!/bin/bash

# Quick start script for running translation tasks with MARBLE

echo "=========================================="
echo "MARBLE Translation Task Runner"
echo "=========================================="

# Check if config file exists
CONFIG_FILE="marble/configs/translation_config_tree.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    echo "Please create a translation config file first."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found."
    echo "Please create a .env file with your API keys:"
    echo "  OPENAI_API_KEY=your_key_here"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run the translation task
echo "Starting translation task..."
echo "Config: $CONFIG_FILE"
echo ""

# python -m marble.main --config_path "$CONFIG_FILE"
poetry run python -m marble.main --config_path "$CONFIG_FILE"


echo ""
echo "=========================================="
echo "Translation task completed!"
echo "Check translation_output.jsonl for results"
echo "=========================================="

