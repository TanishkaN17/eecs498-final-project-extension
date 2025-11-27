# Quick start script for running translation tasks with MARBLE (PowerShell)

Write-Host "=========================================="
Write-Host "MARBLE Translation Task Runner"
Write-Host "=========================================="

# Check if config file exists
$CONFIG_FILE = "marble\configs\translation_config_tree.yaml"

if (-not (Test-Path $CONFIG_FILE)) {
    Write-Host "Error: Config file not found at $CONFIG_FILE"
    Write-Host "Please create a translation config file first."
    exit 1
}

# Check if .env file exists and load it
if (-not (Test-Path ".env")) {
    Write-Host "Warning: .env file not found."
    Write-Host "Please create a .env file with your API keys:"
    Write-Host "  ANTHROPIC_API_KEY=your_key_here"
    Write-Host ""
} else {
    # Load .env file and set environment variables
    Write-Host "Loading environment variables from .env file..."
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]*)\s*=\s*(.*)\s*$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            # Remove comments (everything after #)
            if ($value -match '^([^#]*)#') {
                $value = $matches[1].Trim()
            }
            # Remove quotes if present
            if ($value -match '^["''](.*)["'']$') {
                $value = $matches[1]
            }
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
            Write-Host "  Loaded: $key" -NoNewline
            Write-Host " (value hidden for security)"
        }
    }
    Write-Host ""
}

# Add Poetry to PATH
$POETRY_PATH = "$env:USERPROFILE\AppData\Roaming\Python\Scripts"
if (Test-Path "$POETRY_PATH\poetry.exe") {
    $env:Path = "$POETRY_PATH;$env:Path"
}

# Deactivate any conda/virtual environments that might interfere
if ($env:CONDA_DEFAULT_ENV) {
    Write-Host "Deactivating conda environment: $env:CONDA_DEFAULT_ENV"
    conda deactivate 2>$null
}

# Remove conda from PATH temporarily to avoid conflicts
$env:Path = ($env:Path -split ';' | Where-Object { $_ -notlike '*anaconda3*' -and $_ -notlike '*conda*' }) -join ';'
$env:Path = "$POETRY_PATH;$env:Path"

# Ensure Poetry uses the correct virtual environment (py3.9 with packages)
$VENV_PY39 = "C:\Users\bhava\AppData\Local\pypoetry\Cache\virtualenvs\marble-RUzvaX0V-py3.9\Scripts\python.exe"
if (Test-Path $VENV_PY39) {
    poetry env use $VENV_PY39 2>$null | Out-Null
}

# Set environment variables to prevent Python from using Anaconda's libraries
# This fixes the SSL DLL import issue
$VENV_BASE = "C:\Users\bhava\AppData\Local\pypoetry\Cache\virtualenvs\marble-RUzvaX0V-py3.9"
$VENV_PYTHON = "$VENV_BASE\Scripts\python.exe"
$ANACONDA_BASE = "C:\Users\bhava\anaconda3"

# Clear environment variables that might point to Anaconda
Remove-Item Env:\PYTHONHOME -ErrorAction SilentlyContinue
$env:PYTHONNOUSERSITE = "1"  # Don't use user site-packages
$env:PYTHONPATH = "$VENV_BASE\Lib;$VENV_BASE\Lib\site-packages"  # Use venv's libraries

# Add venv's directories to PATH first, then Anaconda's DLL directory for DLL loading
# The venv Python needs Anaconda's DLLs since it was created from Anaconda Python
$env:Path = "$VENV_BASE\Scripts;$VENV_BASE\DLLs;$VENV_BASE;$ANACONDA_BASE\DLLs;$ANACONDA_BASE\Library\bin;$env:Path"

# Run the translation task directly using the venv Python
# This avoids poetry run which might not properly isolate DLL paths
Write-Host "Starting translation task..."
Write-Host "Config: $CONFIG_FILE"
Write-Host "Using Poetry virtual environment: $VENV_BASE"
Write-Host ""

# Use the venv Python directly with proper environment isolation
& $VENV_PYTHON -m marble.main --config_path $CONFIG_FILE

Write-Host ""
Write-Host "=========================================="
Write-Host "Translation task completed!"
Write-Host "Check translation_output.jsonl for results"
Write-Host "=========================================="