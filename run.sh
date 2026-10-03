#!/usr/bin/env bash
# Sydney Urban Heat & Solar Forecaster — Clean Startup Script
# Automatically unsets any interfering Conda variables and runs the project's .venv

set -e
cd "$(dirname "$0")"

# Clear any active Conda / foreign Python environment variables
unset PYTHONPATH
unset PYTHONHOME
unset LD_LIBRARY_PATH
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV

# Make sure port 8501 is not blocked by a stale process
fuser -k 8501/tcp 2>/dev/null || true

# Run Streamlit using project virtual environment
exec .venv/bin/streamlit run app.py "$@"
