#!/bin/bash

# run_danielson.sh
# This script runs the Danielson-Archer application

# Set script to exit on error
set -e

# Change to the project root directory
cd "$(dirname "$0")/.."

# Load environment variables if .env file exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Check for required environment variables
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Warning: OPENROUTER_API_KEY is not set. Some features may not work properly."
fi

if [ -z "$ARGILLA_API_URL" ]; then
    echo "Warning: ARGILLA_API_URL is not set. Using default (http://localhost:6900)."
    export ARGILLA_API_URL="http://localhost:6900"
fi

if [ -z "$ARGILLA_API_KEY" ]; then
    echo "Warning: ARGILLA_API_KEY is not set. Using default (admin.apikey)."
    export ARGILLA_API_KEY="admin.apikey"
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

# Check for pip and install requirements if necessary
if ! python3 -c "import gradio" &> /dev/null; then
    echo "Installing required packages..."
    python3 -m pip install -r data_labelling/requirements.txt
fi

# Parse command-line arguments
PORT=7860
SHARE=false
NO_ARCHER=false

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --port)
        PORT="$2"
        shift
        shift
        ;;
        --share)
        SHARE=true
        shift
        ;;
        --no-archer)
        NO_ARCHER=true
        shift
        ;;
        *)
        echo "Unknown option: $1"
        echo "Usage: ./run_danielson.sh [--port PORT] [--share] [--no-archer]"
        exit 1
        ;;
    esac
done

# Build the command
CMD="python3 data_labelling/app.py --port $PORT"

if [ "$SHARE" = true ]; then
    CMD="$CMD --share"
fi

if [ "$NO_ARCHER" = true ]; then
    CMD="$CMD --no-archer"
fi

# Run the application
echo "Starting Danielson-Archer application on port $PORT..."
echo "Running: $CMD"
eval $CMD 