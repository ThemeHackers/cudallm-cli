$tools = @("nvcc","ncu","nsys","nvidia-smi")
Write-Host "Checking common CUDA tools..."
foreach ($t in $tools) {
    $path = (Get-Command $t -ErrorAction SilentlyContinue).Path
    if ($path) { Write-Host "Found $t => $path" -ForegroundColor Green }
    else { Write-Host "Missing: $t" -ForegroundColor Yellow }
}


$repoRoot = Split-Path -Parent $PSScriptRoot
$exe = Get-ChildItem -Path $repoRoot -Filter "*llama*server*.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if ($exe) { Write-Host "Found LLM server: $($exe.FullName)" -ForegroundColor Green }
else { Write-Host "LLM server not found in repo root. Please unzip it there." -ForegroundColor Yellow }

Write-Host "Run 'cudallm init' to persist discovered paths into config/config.json" -ForegroundColor Cyan
