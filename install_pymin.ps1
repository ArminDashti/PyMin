# PyMin Installation Script
# This is a comprehensive installer for PyMin

param(
    [switch]$UserOnly = $false,
    [switch]$SkipDependencies = $false,
    [switch]$Force = $false
)

$ErrorActionPreference = "Stop"

# Get the current script directory (PyMin project root)
$PyMinRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PyMinPath = Join-Path $PyMinRoot "PyMin"

Write-Host "PyMin Installation Script" -ForegroundColor Green
Write-Host "=========================" -ForegroundColor Green
Write-Host ""

# Check if running as administrator for system-wide installation
if (-not $UserOnly -and -not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "System-wide installation requires administrator privileges."
    Write-Host "Either run PowerShell as Administrator or use -UserOnly flag for user-level installation."
    Write-Host ""
    $response = Read-Host "Continue with user-level installation? (Y/n)"
    if ($response -eq 'n' -or $response -eq 'N') {
        Write-Host "Installation cancelled."
        exit 0
    }
    $UserOnly = $true
}

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $PythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
    Write-Host "Found Python: $PythonVersion" -ForegroundColor Green
} catch {
    Write-Error "Python is not installed or not in PATH. Please install Python 3.7+ and try again."
    Write-Host "Download Python from: https://www.python.org/downloads/"
    exit 1
}

# Check if PyMin directory exists
if (-not (Test-Path $PyMinPath)) {
    Write-Error "PyMin directory not found at: $PyMinPath"
    Write-Host "Please run this script from the PyMin project root directory."
    exit 1
}

# Install Python dependencies
if (-not $SkipDependencies) {
    Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
    
    # Check for requirements files
    $RequirementsFiles = @(
        (Join-Path $PyMinPath "api\openrouter\requirements.txt"),
        (Join-Path $PyMinRoot "requirements.txt")
    )
    
    $InstalledAny = $false
    foreach ($ReqFile in $RequirementsFiles) {
        if (Test-Path $ReqFile) {
            Write-Host "Installing dependencies from: $ReqFile" -ForegroundColor Cyan
            try {
                pip install -r $ReqFile
                $InstalledAny = $true
            } catch {
                Write-Warning "Failed to install dependencies from $ReqFile : $($_.Exception.Message)"
            }
        }
    }
    
    if (-not $InstalledAny) {
        Write-Host "No requirements.txt files found. Installing basic dependencies..." -ForegroundColor Yellow
        try {
            pip install requests
        } catch {
            Write-Warning "Failed to install basic dependencies: $($_.Exception.Message)"
        }
    }
}

# Run the setup script
Write-Host "Setting up PyMin in PATH..." -ForegroundColor Yellow
try {
    & (Join-Path $PyMinRoot "setup_pymin.ps1") -UserOnly:$UserOnly -Force:$Force
} catch {
    Write-Error "Failed to setup PyMin: $($_.Exception.Message)"
    exit 1
}

# Create desktop shortcut (optional)
Write-Host ""
$CreateShortcut = Read-Host "Create desktop shortcut for PyMin? (y/N)"
if ($CreateShortcut -eq 'y' -or $CreateShortcut -eq 'Y') {
    try {
        $DesktopPath = [Environment]::GetFolderPath("Desktop")
        $ShortcutPath = Join-Path $DesktopPath "PyMin.lnk"
        
        $WshShell = New-Object -ComObject WScript.Shell
        $Shortcut = $WshShell.CreateShortcut($ShortcutPath)
        $Shortcut.TargetPath = "powershell.exe"
        $Shortcut.Arguments = "-NoExit -Command `"cd '$PyMinRoot'; pymin --help`""
        $Shortcut.Description = "PyMin - Python Toolkit"
        $Shortcut.Save()
        
        Write-Host "Desktop shortcut created!" -ForegroundColor Green
    } catch {
        Write-Warning "Could not create desktop shortcut: $($_.Exception.Message)"
    }
}

# Create start menu entry (optional)
Write-Host ""
$CreateStartMenu = Read-Host "Create Start Menu entry for PyMin? (y/N)"
if ($CreateStartMenu -eq 'y' -or $CreateStartMenu -eq 'Y') {
    try {
        if ($UserOnly) {
            $StartMenuPath = [Environment]::GetFolderPath("StartMenu")
        } else {
            $StartMenuPath = [Environment]::GetFolderPath("CommonStartMenu")
        }
        
        $PyMinStartMenuPath = Join-Path $StartMenuPath "PyMin"
        if (-not (Test-Path $PyMinStartMenuPath)) {
            New-Item -ItemType Directory -Path $PyMinStartMenuPath -Force | Out-Null
        }
        
        $ShortcutPath = Join-Path $PyMinStartMenuPath "PyMin.lnk"
        
        $WshShell = New-Object -ComObject WScript.Shell
        $Shortcut = $WshShell.CreateShortcut($ShortcutPath)
        $Shortcut.TargetPath = "powershell.exe"
        $Shortcut.Arguments = "-NoExit -Command `"cd '$PyMinRoot'; pymin --help`""
        $Shortcut.Description = "PyMin - Python Toolkit"
        $Shortcut.Save()
        
        Write-Host "Start Menu entry created!" -ForegroundColor Green
    } catch {
        Write-Warning "Could not create Start Menu entry: $($_.Exception.Message)"
    }
}

Write-Host ""
Write-Host "Installation completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "PyMin is now available as a command in PowerShell!" -ForegroundColor Cyan
Write-Host ""
Write-Host "Quick start:" -ForegroundColor Yellow
Write-Host "1. Close and reopen your PowerShell/Command Prompt" -ForegroundColor White
Write-Host "2. Run: pymin --help" -ForegroundColor White
Write-Host ""
Write-Host "For more information, visit the PyMin project directory:" -ForegroundColor Cyan
Write-Host $PyMinRoot -ForegroundColor White
