#!/bin/bash
# Touchless Docker Dashboard Runner Script

# Resolve directory path of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change directory to script's directory so relative paths are resolved correctly
cd "$SCRIPT_DIR"

# Execute main.py using virtual environment python
"./venv/bin/python" "main.py"
