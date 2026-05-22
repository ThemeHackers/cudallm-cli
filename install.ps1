Write-Host "==================================================" -ForegroundColor Green
Write-Host "Starting automated installation for cudallm-cli" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green

if ($null -eq $env:VIRTUAL_ENV) {
    if (Test-Path ".venv") {
        Write-Host "[INFO] Activating existing virtual environment..." -ForegroundColor Cyan
        . .venv\Scripts\Activate.ps1
    } else {
        Write-Host "[INFO] Creating new virtual environment (.venv)..." -ForegroundColor Cyan
        python -m venv .venv
        . .venv\Scripts\Activate.ps1
    }
}

Write-Host "[STEP 1/3] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

Write-Host "[STEP 2/3] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt --no-cache-dir

Write-Host "[STEP 3/3] Installing cudallm-cli in editable mode..." -ForegroundColor Yellow
pip install -e .

Write-Host "[INFO] Initializing environment config..." -ForegroundColor Cyan
cudallm init

Write-Host "==================================================" -ForegroundColor Green
Write-Host "Installation completed! Run 'cudallm doctor' to verify." -ForegroundColor Green
Write-Host "Please ensure you run '.venv\Scripts\Activate.ps1' in your shell." -ForegroundColor Yellow
Write-Host "==================================================" -ForegroundColor Green
