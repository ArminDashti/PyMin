"""
IP Address Utilities

This module provides functions to detect internal and external IP addresses.
"""

import socket
import requests
import subprocess
import platform
import re
from typing import List, Dict, Optional, Tuple


def get_internal_ip() -> Optional[str]:
    """
    Get the current internal IP address of the machine.
    
    Returns:
        str: Internal IP address, or None if not found
    """
    try:
        # Method 1: Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Connect to a remote address (doesn't actually send data)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            return local_ip
    except Exception:
        pass
    
    try:
        # Method 2: Use hostname resolution
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        if local_ip != "127.0.0.1":
            return local_ip
    except Exception:
        pass
    
    try:
        # Method 3: Platform-specific commands
        system = platform.system().lower()
        
        if system == "windows":
            result = subprocess.run(
                ["ipconfig"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                # Parse IPv4 addresses from ipconfig output
                ip_pattern = r'IPv4 Address[.\s]*:\s*(\d+\.\d+\.\d+\.\d+)'
                matches = re.findall(ip_pattern, result.stdout)
                for ip in matches:
                    if not ip.startswith("127.") and not ip.startswith("169.254."):
                        return ip
        
        elif system in ["linux", "darwin"]:  # Linux or macOS
            result = subprocess.run(
                ["hostname", "-I"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                ips = result.stdout.strip().split()
                for ip in ips:
                    if not ip.startswith("127.") and not ip.startswith("169.254."):
                        return ip
    except Exception:
        pass
    
    return None


def get_external_ip() -> Optional[str]:
    """
    Get the current external/public IP address of the machine.
    
    Returns:
        str: External IP address, or None if not found
    """
    # List of IP checking services
    ip_services = [
        "https://api.ipify.org",
        "https://ipv4.icanhazip.com",
        "https://ident.me",
        "https://api.my-ip.io/ip",
        "https://ipinfo.io/ip",
        "https://checkip.amazonaws.com"
    ]
    
    for service in ip_services:
        try:
            response = requests.get(service, timeout=10)
            if response.status_code == 200:
                ip = response.text.strip()
                # Validate IP format
                if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', ip):
                    return ip
        except Exception:
            continue
    
    return None


def get_all_ips() -> Dict[str, List[str]]:
    """
    Get all available IP addresses (internal and external).
    
    Returns:
        dict: Dictionary containing 'internal', 'external', and 'all' IP lists
    """
    result = {
        'internal': [],
        'external': [],
        'all': []
    }
    
    # Get internal IP
    internal_ip = get_internal_ip()
    if internal_ip:
        result['internal'].append(internal_ip)
        result['all'].append(internal_ip)
    
    # Get external IP
    external_ip = get_external_ip()
    if external_ip:
        result['external'].append(external_ip)
        if external_ip not in result['all']:
            result['all'].append(external_ip)
    
    # Get additional internal IPs
    try:
        system = platform.system().lower()
        
        if system == "windows":
            result_cmd = subprocess.run(
                ["ipconfig"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result_cmd.returncode == 0:
                ip_pattern = r'IPv4 Address[.\s]*:\s*(\d+\.\d+\.\d+\.\d+)'
                matches = re.findall(ip_pattern, result_cmd.stdout)
                for ip in matches:
                    if ip not in result['internal'] and not ip.startswith("127."):
                        result['internal'].append(ip)
                        if ip not in result['all']:
                            result['all'].append(ip)
        
        elif system in ["linux", "darwin"]:
            result_cmd = subprocess.run(
                ["hostname", "-I"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result_cmd.returncode == 0:
                ips = result_cmd.stdout.strip().split()
                for ip in ips:
                    if ip not in result['internal'] and not ip.startswith("127."):
                        result['internal'].append(ip)
                        if ip not in result['all']:
                            result['all'].append(ip)
    except Exception:
        pass
    
    return result


def get_ip_info(ip_address: str) -> Optional[Dict]:
    """
    Get detailed information about an IP address.
    
    Args:
        ip_address (str): IP address to look up
        
    Returns:
        dict: IP information including location, ISP, etc.
    """
    try:
        response = requests.get(f"http://ip-api.com/json/{ip_address}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    
    return None


def is_private_ip(ip_address: str) -> bool:
    """
    Check if an IP address is private.
    
    Args:
        ip_address (str): IP address to check
        
    Returns:
        bool: True if private, False otherwise
    """
    try:
        import ipaddress
        ip = ipaddress.ip_address(ip_address)
        return ip.is_private
    except Exception:
        # Fallback regex check
        private_patterns = [
            r'^10\.',
            r'^172\.(1[6-9]|2[0-9]|3[0-1])\.',
            r'^192\.168\.',
            r'^127\.',
            r'^169\.254\.'
        ]
        return any(re.match(pattern, ip_address) for pattern in private_patterns)


def get_network_interfaces() -> List[Dict]:
    """
    Get information about network interfaces.
    
    Returns:
        list: List of dictionaries containing interface information
    """
    interfaces = []
    
    try:
        system = platform.system().lower()
        
        if system == "windows":
            result = subprocess.run(
                ["ipconfig", "/all"], 
                capture_output=True, 
                text=True, 
                timeout=15
            )
            if result.returncode == 0:
                # Parse network adapter information
                current_adapter = {}
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if 'adapter' in line.lower() and ':' in line:
                        if current_adapter:
                            interfaces.append(current_adapter)
                        current_adapter = {'name': line.split(':')[0].strip()}
                    elif 'IPv4 Address' in line and ':' in line:
                        ip = line.split(':')[1].strip()
                        if ip and ip != '(Preferred)':
                            current_adapter['ipv4'] = ip
                    elif 'Subnet Mask' in line and ':' in line:
                        mask = line.split(':')[1].strip()
                        current_adapter['subnet_mask'] = mask
                    elif 'Default Gateway' in line and ':' in line:
                        gateway = line.split(':')[1].strip()
                        current_adapter['gateway'] = gateway
                
                if current_adapter:
                    interfaces.append(current_adapter)
        
        elif system in ["linux", "darwin"]:
            result = subprocess.run(
                ["ifconfig"], 
                capture_output=True, 
                text=True, 
                timeout=15
            )
            if result.returncode == 0:
                # Parse ifconfig output
                current_interface = {}
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if line and not line.startswith(' ') and not line.startswith('\t'):
                        if current_interface:
                            interfaces.append(current_interface)
                        current_interface = {'name': line.split(':')[0]}
                    elif 'inet ' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'inet' and i + 1 < len(parts):
                                current_interface['ipv4'] = parts[i + 1]
                                break
                    elif 'netmask' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'netmask' and i + 1 < len(parts):
                                current_interface['subnet_mask'] = parts[i + 1]
                                break
                
                if current_interface:
                    interfaces.append(current_interface)
    
    except Exception:
        pass
    
    return interfaces
