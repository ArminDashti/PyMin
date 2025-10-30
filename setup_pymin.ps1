# PyMin Setup Script for PowerShell
# This script adds PyMin to the system PATH environment variable

param(
    [switch]$UserOnly = $false,
    [switch]$Force = $false
)

# Get the current script directory (PyMin project root)
$PyMinRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PyMinPath = Join-Path $PyMinRoot "PyMin"

Write-Host "PyMin Setup Script" -ForegroundColor Green
Write-Host "=================" -ForegroundColor Green
Write-Host ""

# Check if PyMin directory exists
if (-not (Test-Path $PyMinPath)) {
    Write-Error "PyMin directory not found at: $PyMinPath"
    Write-Host "Please run this script from the PyMin project root directory."
    exit 1
}

# Check if pymin.py exists
$PyMinScript = Join-Path $PyMinRoot "pymin.py"
if (-not (Test-Path $PyMinScript)) {
    Write-Error "pymin.py not found at: $PyMinScript"
    exit 1
}

# Determine scope for environment variable
if ($UserOnly) {
    $Scope = "User"
    Write-Host "Setting up PyMin for current user only..." -ForegroundColor Yellow
} else {
    $Scope = "Machine"
    Write-Host "Setting up PyMin for all users (requires admin privileges)..." -ForegroundColor Yellow
}

# Get current PATH
try {
    $CurrentPath = [Environment]::GetEnvironmentVariable("PATH", $Scope)
} catch {
    Write-Error "Failed to access PATH environment variable. $($_.Exception.Message)"
    if (-not $UserOnly) {
        Write-Host "Try running as administrator or use -UserOnly flag for user-level installation."
    }
    exit 1
}

# Check if PyMin is already in PATH
if ($CurrentPath -like "*$PyMinRoot*") {
    if (-not $Force) {
        Write-Warning "PyMin appears to already be in the PATH environment variable."
        Write-Host "Current PATH entries containing PyMin:"
        $CurrentPath -split ';' | Where-Object { $_ -like "*PyMin*" } | ForEach-Object { Write-Host "  $_" }
        Write-Host ""
        $response = Read-Host "Do you want to continue anyway? (y/N)"
        if ($response -ne 'y' -and $response -ne 'Y') {
            Write-Host "Setup cancelled."
            exit 0
        }
    }
    Write-Host "Removing existing PyMin entries from PATH..." -ForegroundColor Yellow
    $NewPath = ($CurrentPath -split ';' | Where-Object { $_ -notlike "*PyMin*" }) -join ';'
} else {
    $NewPath = $CurrentPath
}

# Add PyMin to PATH
if ($NewPath.EndsWith(';')) {
    $NewPath += $PyMinRoot
} else {
    $NewPath += ";" + $PyMinRoot
}

# Set the new PATH
try {
    [Environment]::SetEnvironmentVariable("PATH", $NewPath, $Scope)
    Write-Host "Successfully added PyMin to PATH!" -ForegroundColor Green
    Write-Host "PyMin root directory: $PyMinRoot" -ForegroundColor Cyan
} catch {
    Write-Error "Failed to update PATH environment variable: $($_.Exception.Message)"
    exit 1
}

# Create a batch file for easier access
$BatchFile = Join-Path $PyMinRoot "pymin.bat"
$BatchContent = @"
@echo off
python "$PyMinScript" %*
"@

try {
    $BatchContent | Out-File -FilePath $BatchFile -Encoding ASCII
    Write-Host "Created pymin.bat for easier access" -ForegroundColor Green
} catch {
    Write-Warning "Could not create pymin.bat: $($_.Exception.Message)"
}

Write-Host ""
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "To use PyMin:" -ForegroundColor Cyan
Write-Host "1. Close and reopen your PowerShell/Command Prompt" -ForegroundColor White
Write-Host "2. Run: pymin --help" -ForegroundColor White
Write-Host ""
Write-Host "Examples:" -ForegroundColor Cyan
Write-Host "  pymin openrouter --message 'Hello, world!'" -ForegroundColor White
Write-Host "  pymin db mssql --server localhost --database test --query 'SELECT * FROM users'" -ForegroundColor White
Write-Host "  pymin image --input photo.jpg --output photo.png" -ForegroundColor White
Write-Host ""

# Test the installation
Write-Host "Testing PyMin installation..." -ForegroundColor Yellow
try {
    $TestResult = & python $PyMinScript --help 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "PyMin is working correctly!" -ForegroundColor Green
    } else {
        Write-Warning "PyMin test failed. You may need to restart your terminal."
    }
} catch {
    Write-Warning "Could not test PyMin: $($_.Exception.Message)"
    Write-Host "Try restarting your terminal and running: pymin --help"
}
