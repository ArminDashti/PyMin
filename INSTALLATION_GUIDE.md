# PyMin Installation Guide

This guide provides multiple ways to install PyMin on your PC.

## Prerequisites

- Python 3.7 or higher
- pip (usually comes with Python)
- Windows 10 or higher (for PowerShell scripts)

## Installation Methods

### Method 1: pip Install (Recommended)

This is the standard Python package installation method:

```bash
# Navigate to the PyMin directory
cd path/to/PyMin

# Install PyMin
pip install .

# Or install in development mode (for developers)
pip install -e .
```

After installation, you can use PyMin from anywhere:
```bash
pymin --help
```

### Method 2: PowerShell Script (Windows)

For Windows users who prefer automated installation:

```powershell
# Run as Administrator for system-wide installation
.\install_pymin.ps1

# Or for user-only installation (no admin required)
.\install_pymin.ps1 -UserOnly
```

### Method 3: Manual Installation

1. **Clone or download PyMin**:
   ```bash
   git clone https://github.com/yourusername/PyMin.git
   cd PyMin
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add to PATH** (Windows):
   ```powershell
   # Add PyMin directory to PATH
   $env:PATH += ";C:\path\to\PyMin"
   
   # Or add permanently to user PATH
   [Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\path\to\PyMin", "User")
   ```

4. **Test installation**:
   ```bash
   python test_installation.py
   ```

## Verification

After installation, verify PyMin is working:

```bash
# Test the CLI
pymin --help

# Test with a simple command
pymin version

# Run the test script
python test_installation.py
```

## Uninstallation

### Via pip
```bash
pip uninstall pymin
```

### Via PowerShell (Windows)
```powershell
.\uninstall_pymin.ps1
```

### Manual
- Remove PyMin directory from PATH
- Delete the PyMin project folder

## Troubleshooting

### "pymin command not found"
- Restart your terminal/command prompt
- Check if PyMin is in your PATH: `echo $env:PATH` (Windows) or `echo $PATH` (Linux/Mac)
- Try running: `python -m PyMin --help`

### Permission errors
- Run PowerShell as Administrator for system-wide installation
- Or use the `-UserOnly` flag for user-level installation

### Python not found
- Install Python from https://www.python.org/downloads/
- Make sure Python is added to your PATH during installation

### Import errors
- Install missing dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.7+)

## Development Installation

For developers who want to modify PyMin:

```bash
# Clone the repository
git clone https://github.com/yourusername/PyMin.git
cd PyMin

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements.txt
```

## System Requirements

- **Python**: 3.7 or higher
- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Disk Space**: 500MB for installation

## Next Steps

After successful installation:

1. Read the [README.md](README.md) for usage examples
2. Check [PYMIN_SKLEARN_GUIDE.md](PYMIN_SKLEARN_GUIDE.md) for machine learning features
3. Explore the [SKLEARN_USAGE.md](SKLEARN_USAGE.md) for scikit-learn integration
4. Try the examples in the documentation

## Support

If you encounter issues:

1. Check this installation guide
2. Review the troubleshooting section
3. Check the [INSTALL.md](INSTALL.md) for additional details
4. Open an issue on GitHub with:
   - Your operating system
   - Python version (`python --version`)
   - Error messages
   - Installation method used
