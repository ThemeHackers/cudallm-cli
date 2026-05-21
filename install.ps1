
Write-Host "==================================================" -ForegroundColor Green
Write-Host "Starting automated CUDA installation for cudallm-cli" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green


if ($null -eq $env:VIRTUAL_ENV) {
    if (Test-Path ".venv") {
        Write-Host "Activating existing virtual environment..." -ForegroundColor Yellow
        . .venv\Scripts\Activate.ps1
    } else {
        Write-Host "Creating new virtual environment (.venv)..." -ForegroundColor Yellow
        python -m venv .venv
        . .venv\Scripts\Activate.ps1
    }
}


python -m pip install --upgrade pip



Write-Host "Installing dependencies from requirements.txt with CUDA support..." -ForegroundColor Yellow
pip install -r requirements.txt --no-cache-dir

Write-Host "Installing cudallm-cli in editable mode..." -ForegroundColor Yellow
pip install -e .


Write-Host "Initializing environment..." -ForegroundColor Yellow
cudallm init

Write-Host "==================================================" -ForegroundColor Green
Write-Host "Installation completed! Run 'cudallm doctor' to verify." -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
