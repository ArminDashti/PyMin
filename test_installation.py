#!/usr/bin/env python3
"""
Test script to verify PyMin installation
"""

import sys
import subprocess
import importlib

def test_imports():
    """Test that all main modules can be imported"""
    print("Testing module imports...")
    
    modules_to_test = [
        "PyMin",
        "PyMin.api.openrouter.openrouter_client",
        "PyMin.classification.base_classifier",
        "PyMin.regression.base",
        "PyMin.timeseries.prophet_forecaster",
        "PyMin.network.dns_utils",
        "PyMin.db.mssql.mssql",
        "PyMin.util.image_converter",
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_cli():
    """Test that the CLI command works"""
    print("\nTesting CLI command...")
    
    try:
        result = subprocess.run([sys.executable, "-m", "PyMin", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ CLI command works")
            return True
        else:
            print(f"✗ CLI command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ CLI command error: {e}")
        return False

def test_dependencies():
    """Test that key dependencies are available"""
    print("\nTesting dependencies...")
    
    dependencies = [
        "requests",
        "pandas", 
        "numpy",
        "sklearn",
        "matplotlib",
        "seaborn",
    ]
    
    failed_deps = []
    
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep}")
            failed_deps.append(dep)
    
    return len(failed_deps) == 0

def main():
    """Run all tests"""
    print("PyMin Installation Test")
    print("=" * 30)
    
    tests = [
        ("Module Imports", test_imports),
        ("CLI Command", test_cli),
        ("Dependencies", test_dependencies),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n{'=' * 30}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! PyMin is properly installed.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
