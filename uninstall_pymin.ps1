# PyMin Uninstallation Script
# This script removes PyMin from the system PATH and cleans up files

param(
    [switch]$UserOnly = $false,
    [switch]$Force = $false
)

Write-Host "PyMin Uninstallation Script" -ForegroundColor Red
Write-Host "============================" -ForegroundColor Red
Write-Host ""

# Get the current script directory (PyMin project root)
$PyMinRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

# Determine scope for environment variable
if ($UserOnly) {
    $Scope = "User"
    Write-Host "Removing PyMin for current user only..." -ForegroundColor Yellow
} else {
    $Scope = "Machine"
    Write-Host "Removing PyMin for all users (requires admin privileges)..." -ForegroundColor Yellow
}

# Check if running as administrator for system-wide uninstallation
if (-not $UserOnly -and -not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "System-wide uninstallation requires administrator privileges."
    Write-Host "Either run PowerShell as Administrator or use -UserOnly flag for user-level uninstallation."
    Write-Host ""
    $response = Read-Host "Continue with user-level uninstallation? (Y/n)"
    if ($response -eq 'n' -or $response -eq 'N') {
        Write-Host "Uninstallation cancelled."
        exit 0
    }
    $UserOnly = $true
}

# Get current PATH
try {
    $CurrentPath = [Environment]::GetEnvironmentVariable("PATH", $Scope)
} catch {
    Write-Error "Failed to access PATH environment variable. $($_.Exception.Message)"
    if (-not $UserOnly) {
        Write-Host "Try running as administrator or use -UserOnly flag for user-level uninstallation."
    }
    exit 1
}

# Check if PyMin is in PATH
if ($CurrentPath -notlike "*$PyMinRoot*") {
    Write-Warning "PyMin does not appear to be in the PATH environment variable."
    if (-not $Force) {
        $response = Read-Host "Continue with cleanup anyway? (y/N)"
        if ($response -ne 'y' -and $response -ne 'Y') {
            Write-Host "Uninstallation cancelled."
            exit 0
        }
    }
} else {
    Write-Host "Removing PyMin from PATH..." -ForegroundColor Yellow
    $NewPath = ($CurrentPath -split ';' | Where-Object { $_ -notlike "*PyMin*" }) -join ';'
    
    # Set the new PATH
    try {
        [Environment]::SetEnvironmentVariable("PATH", $NewPath, $Scope)
        Write-Host "Successfully removed PyMin from PATH!" -ForegroundColor Green
    } catch {
        Write-Error "Failed to update PATH environment variable: $($_.Exception.Message)"
        exit 1
    }
}

# Remove batch file
$BatchFile = Join-Path $PyMinRoot "pymin.bat"
if (Test-Path $BatchFile) {
    try {
        Remove-Item $BatchFile -Force
        Write-Host "Removed pymin.bat" -ForegroundColor Green
    } catch {
        Write-Warning "Could not remove pymin.bat: $($_.Exception.Message)"
    }
}

# Remove desktop shortcut
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $DesktopPath "PyMin.lnk"
if (Test-Path $ShortcutPath) {
    try {
        Remove-Item $ShortcutPath -Force
        Write-Host "Removed desktop shortcut" -ForegroundColor Green
    } catch {
        Write-Warning "Could not remove desktop shortcut: $($_.Exception.Message)"
    }
}

# Remove start menu entries
$StartMenuPaths = @(
    [Environment]::GetFolderPath("StartMenu"),
    [Environment]::GetFolderPath("CommonStartMenu")
)

foreach ($StartMenuPath in $StartMenuPaths) {
    $PyMinStartMenuPath = Join-Path $StartMenuPath "PyMin"
    if (Test-Path $PyMinStartMenuPath) {
        try {
            Remove-Item $PyMinStartMenuPath -Recurse -Force
            Write-Host "Removed Start Menu entry: $PyMinStartMenuPath" -ForegroundColor Green
        } catch {
            Write-Warning "Could not remove Start Menu entry: $($_.Exception.Message)"
        }
    }
}

Write-Host ""
Write-Host "Uninstallation completed!" -ForegroundColor Green
Write-Host ""
Write-Host "PyMin has been removed from your system." -ForegroundColor Cyan
Write-Host "Note: The PyMin project files in $PyMinRoot have not been deleted." -ForegroundColor Yellow
Write-Host "You can manually delete the project directory if you no longer need it." -ForegroundColor Yellow
