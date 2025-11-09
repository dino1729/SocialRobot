# Windows Setup Script for SocialRobot
# Run this in PowerShell on Windows

Write-Host "Setting up SocialRobot on Windows..." -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate and install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Copy .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    if (Test-Path "env.example") {
        Copy-Item "env.example" ".env"
        Write-Host "✓ Created .env from env.example - Please configure it!" -ForegroundColor Yellow
    } else {
        Write-Host "⚠ No .env or env.example found - you'll need to create one" -ForegroundColor Yellow
    }
}

Write-Host "`n✓ Setup complete!" -ForegroundColor Green
Write-Host "To run the project:" -ForegroundColor Cyan
Write-Host "  1. Activate: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "  2. Run: python main.py" -ForegroundColor Cyan

