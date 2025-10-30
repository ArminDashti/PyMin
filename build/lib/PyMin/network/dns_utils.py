"""
DNS Utilities

This module provides functions for DNS configuration detection and leak testing.
"""

import socket
import subprocess
import platform
import requests
import dns.resolver
import dns.exception
from typing import List, Dict, Optional, Tuple
import re
import json


def get_dns_servers() -> List[str]:
    """
    Get the current DNS servers configured on the system.
    
    Returns:
        list: List of DNS server IP addresses
    """
    dns_servers = []
    
    try:
        system = platform.system().lower()
        
        if system == "windows":
            # Get DNS servers from Windows
            result = subprocess.run(
                ["nslookup", "google.com"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Server:' in line:
                        server = line.split('Server:')[1].strip()
                        if server and server not in dns_servers:
                            dns_servers.append(server)
            
            # Alternative method using ipconfig
            result = subprocess.run(
                ["ipconfig", "/all"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                dns_pattern = r'DNS Servers[.\s]*:\s*(\d+\.\d+\.\d+\.\d+)'
                matches = re.findall(dns_pattern, result.stdout)
                for match in matches:
                    if match not in dns_servers:
                        dns_servers.append(match)
        
        elif system in ["linux", "darwin"]:
            # Try /etc/resolv.conf first
            try:
                with open('/etc/resolv.conf', 'r') as f:
                    content = f.read()
                    nameserver_pattern = r'nameserver\s+(\d+\.\d+\.\d+\.\d+)'
                    matches = re.findall(nameserver_pattern, content)
                    for match in matches:
                        if match not in dns_servers:
                            dns_servers.append(match)
            except Exception:
                pass
            
            # Try systemd-resolved on Linux
            if system == "linux":
                try:
                    result = subprocess.run(
                        ["systemd-resolve", "--status"], 
                        capture_output=True, 
                        text=True, 
                        timeout=10
                    )
                    if result.returncode == 0:
                        dns_pattern = r'DNS Servers: (.+)'
                        match = re.search(dns_pattern, result.stdout)
                        if match:
                            servers = match.group(1).split()
                            for server in servers:
                                if server not in dns_servers:
                                    dns_servers.append(server)
                except Exception:
                    pass
    
    except Exception:
        pass
    
    # Fallback: try to get DNS from resolver
    try:
        resolver = dns.resolver.Resolver()
        if resolver.nameservers:
            for ns in resolver.nameservers:
                ns_str = str(ns)
                if ns_str not in dns_servers:
                    dns_servers.append(ns_str)
    except Exception:
        pass
    
    return dns_servers


def test_dns_resolution(domain: str, dns_server: Optional[str] = None) -> Dict:
    """
    Test DNS resolution for a domain.
    
    Args:
        domain (str): Domain to resolve
        dns_server (str, optional): Specific DNS server to use
        
    Returns:
        dict: Resolution results including IPs and response time
    """
    result = {
        'domain': domain,
        'dns_server': dns_server,
        'ips': [],
        'response_time': None,
        'success': False,
        'error': None
    }
    
    try:
        import time
        start_time = time.time()
        
        if dns_server:
            # Use specific DNS server
            resolver = dns.resolver.Resolver()
            resolver.nameservers = [dns_server]
            answers = resolver.resolve(domain, 'A')
        else:
            # Use system default
            answers = dns.resolver.resolve(domain, 'A')
        
        end_time = time.time()
        result['response_time'] = round((end_time - start_time) * 1000, 2)  # ms
        
        for answer in answers:
            result['ips'].append(str(answer))
        
        result['success'] = True
    
    except dns.exception.DNSException as e:
        result['error'] = str(e)
    except Exception as e:
        result['error'] = str(e)
    
    return result


def check_dns_leak() -> Dict:
    """
    Check for DNS leaks by testing resolution through different DNS servers.
    
    Returns:
        dict: DNS leak test results
    """
    result = {
        'leak_detected': False,
        'configured_dns': [],
        'resolved_ips': [],
        'test_results': [],
        'recommendations': []
    }
    
    # Get configured DNS servers
    configured_dns = get_dns_servers()
    result['configured_dns'] = configured_dns
    
    # Test domains for leak detection
    test_domains = [
        'whoami.akamai.net',
        'whoami.ultradns.net',
        'resolver-address.ultradns.net'
    ]
    
    # Test with configured DNS
    for domain in test_domains:
        test_result = test_dns_resolution(domain)
        result['test_results'].append(test_result)
        
        if test_result['success'] and test_result['ips']:
            for ip in test_result['ips']:
                if ip not in result['resolved_ips']:
                    result['resolved_ips'].append(ip)
    
    # Test with known public DNS servers
    public_dns_servers = [
        '8.8.8.8',      # Google DNS
        '1.1.1.1',      # Cloudflare DNS
        '208.67.222.222', # OpenDNS
        '9.9.9.9'       # Quad9 DNS
    ]
    
    for dns_server in public_dns_servers:
        for domain in test_domains:
            test_result = test_dns_resolution(domain, dns_server)
            test_result['dns_server'] = dns_server
            result['test_results'].append(test_result)
    
    # Analyze results for leaks
    unique_ips = set(result['resolved_ips'])
    if len(unique_ips) > 1:
        result['leak_detected'] = True
        result['recommendations'].append("Multiple different IPs detected - possible DNS leak")
    
    # Check if using VPN/DNS provider
    vpn_indicators = [
        '10.', '172.', '192.168.',  # Private ranges
        '127.'  # Localhost
    ]
    
    for ip in result['resolved_ips']:
        if any(ip.startswith(indicator) for indicator in vpn_indicators):
            result['recommendations'].append(f"Using private IP {ip} - may indicate VPN usage")
    
    return result


def get_dns_over_https_servers() -> List[Dict]:
    """
    Get list of DNS over HTTPS (DoH) servers.
    
    Returns:
        list: List of DoH server configurations
    """
    doh_servers = [
        {
            'name': 'Cloudflare',
            'url': 'https://cloudflare-dns.com/dns-query',
            'ip': '1.1.1.1'
        },
        {
            'name': 'Google',
            'url': 'https://dns.google/dns-query',
            'ip': '8.8.8.8'
        },
        {
            'name': 'Quad9',
            'url': 'https://dns.quad9.net/dns-query',
            'ip': '9.9.9.9'
        },
        {
            'name': 'OpenDNS',
            'url': 'https://doh.opendns.com/dns-query',
            'ip': '208.67.222.222'
        }
    ]
    
    return doh_servers


def test_dns_over_https(domain: str, doh_server: str) -> Dict:
    """
    Test DNS resolution using DNS over HTTPS.
    
    Args:
        domain (str): Domain to resolve
        doh_server (str): DoH server URL
        
    Returns:
        dict: DoH resolution results
    """
    result = {
        'domain': domain,
        'doh_server': doh_server,
        'ips': [],
        'response_time': None,
        'success': False,
        'error': None
    }
    
    try:
        import time
        start_time = time.time()
        
        # Prepare DoH request
        params = {
            'name': domain,
            'type': 'A'
        }
        
        headers = {
            'Accept': 'application/dns-json'
        }
        
        response = requests.get(doh_server, params=params, headers=headers, timeout=10)
        end_time = time.time()
        result['response_time'] = round((end_time - start_time) * 1000, 2)  # ms
        
        if response.status_code == 200:
            data = response.json()
            if 'Answer' in data:
                for answer in data['Answer']:
                    if answer.get('type') == 1:  # A record
                        result['ips'].append(answer['data'])
                result['success'] = True
            else:
                result['error'] = "No answer in response"
        else:
            result['error'] = f"HTTP {response.status_code}"
    
    except Exception as e:
        result['error'] = str(e)
    
    return result


def get_dns_cache() -> List[Dict]:
    """
    Get DNS cache information (Windows only).
    
    Returns:
        list: DNS cache entries
    """
    cache_entries = []
    
    try:
        if platform.system().lower() == "windows":
            result = subprocess.run(
                ["ipconfig", "/displaydns"], 
                capture_output=True, 
                text=True, 
                timeout=15
            )
            if result.returncode == 0:
                # Parse DNS cache entries
                current_entry = {}
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if line.startswith('Record Name'):
                        if current_entry:
                            cache_entries.append(current_entry)
                        current_entry = {}
                    elif ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        value = value.strip()
                        current_entry[key] = value
                
                if current_entry:
                    cache_entries.append(current_entry)
    
    except Exception:
        pass
    
    return cache_entries


def flush_dns_cache() -> bool:
    """
    Flush DNS cache.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        system = platform.system().lower()
        
        if system == "windows":
            result = subprocess.run(
                ["ipconfig", "/flushdns"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        
        elif system == "linux":
            # Try systemd-resolved
            result = subprocess.run(
                ["sudo", "systemd-resolve", "--flush-caches"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                return True
            
            # Try nscd
            result = subprocess.run(
                ["sudo", "nscd", "-i", "hosts"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        
        elif system == "darwin":  # macOS
            result = subprocess.run(
                ["sudo", "dscacheutil", "-flushcache"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
    
    except Exception:
        pass
    
    return False
