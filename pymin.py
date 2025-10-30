#!/usr/bin/env python3
"""
PyMin - Entry point script for command line usage
This script allows PyMin to be used as 'pymin' command in PowerShell
"""

import sys
from pathlib import Path

# Add the PyMin directory to the Python path
pymin_dir = Path(__file__).parent / "PyMin"
sys.path.insert(0, str(pymin_dir))

# Import and run the main function
from __main__ import main

if __name__ == "__main__":
    main()
