# PyMin Installation Guide

PyMin is now set up to be used as a command-line tool in PowerShell. Here's how to install and use it.

## Quick Installation

### Option 1: Full Installation (Recommended)
```powershell
# Run as Administrator for system-wide installation
.\install_pymin.ps1

# Or for user-only installation
.\install_pymin.ps1 -UserOnly
```

### Option 2: Basic Setup Only
```powershell
# Just add to PATH without installing dependencies
.\setup_pymin.ps1

# For user-only installation
.\setup_pymin.ps1 -UserOnly
```

## Usage

After installation, you can use PyMin from any PowerShell window:

```powershell
# Show help
pymin --help

# OpenRouter API
pymin openrouter --message "Hello, world!"

# Database operations
pymin db mssql --server localhost --database test --query "SELECT * FROM users"

# Image conversion
pymin image --input photo.jpg --output photo.png

# Show version
pymin version
```

## Uninstallation

To remove PyMin from your system:

```powershell
# Remove from PATH and clean up shortcuts
.\uninstall_pymin.ps1

# For user-only uninstallation
.\uninstall_pymin.ps1 -UserOnly
```

## Manual Installation

If you prefer to set up PyMin manually:

1. Add the PyMin project root directory to your PATH environment variable
2. Ensure Python is installed and accessible
3. Install required dependencies:
   ```powershell
   pip install requests
   ```

## Requirements

- Python 3.7 or higher
- PowerShell 5.1 or higher
- Windows 10 or higher

## Troubleshooting

### PyMin command not found
- Restart your PowerShell/Command Prompt after installation
- Check if the PyMin directory is in your PATH: `echo $env:PATH`
- Try running the setup script again

### Permission errors
- Run PowerShell as Administrator for system-wide installation
- Or use the `-UserOnly` flag for user-level installation

### Python not found
- Install Python from https://www.python.org/downloads/
- Make sure Python is added to your PATH during installation
