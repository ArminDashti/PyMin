"""
WebRTC Utilities

This module provides functions for WebRTC leak testing and IP detection.
"""

import socket
import requests
import json
import subprocess
import platform
from typing import List, Dict, Optional, Tuple
import re
import threading
import time


def get_webrtc_ips() -> Dict[str, List[str]]:
    """
    Get IP addresses that could be leaked through WebRTC.
    
    Returns:
        dict: Dictionary containing different types of IPs that could be leaked
    """
    result = {
        'local_ips': [],
        'public_ips': [],
        'candidate_ips': [],
        'stun_ips': []
    }
    
    # Get local network interfaces
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
                    if not ip.startswith("127.") and not ip.startswith("169.254."):
                        result['local_ips'].append(ip)
        
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
                    if not ip.startswith("127.") and not ip.startswith("169.254."):
                        result['local_ips'].append(ip)
    except Exception:
        pass
    
    # Get public IP
    try:
        response = requests.get("https://api.ipify.org", timeout=10)
        if response.status_code == 200:
            public_ip = response.text.strip()
            if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', public_ip):
                result['public_ips'].append(public_ip)
    except Exception:
        pass
    
    # Get STUN server IPs (these could be leaked)
    stun_servers = [
        "stun.l.google.com",
        "stun1.l.google.com", 
        "stun2.l.google.com",
        "stun3.l.google.com",
        "stun4.l.google.com",
        "stun.stunprotocol.org",
        "stun.voiparound.com",
        "stun.voipbuster.com"
    ]
    
    for stun_server in stun_servers:
        try:
            ip = socket.gethostbyname(stun_server)
            if ip not in result['stun_ips']:
                result['stun_ips'].append(ip)
        except Exception:
            continue
    
    return result


def check_webrtc_leak() -> Dict:
    """
    Check for WebRTC leaks by simulating WebRTC behavior.
    
    Returns:
        dict: WebRTC leak test results
    """
    result = {
        'leak_detected': False,
        'local_ips_exposed': [],
        'public_ips_exposed': [],
        'stun_servers_accessible': [],
        'ice_candidates': [],
        'recommendations': [],
        'risk_level': 'low'
    }
    
    # Get potential leakable IPs
    webrtc_ips = get_webrtc_ips()
    
    # Check local IP exposure
    result['local_ips_exposed'] = webrtc_ips['local_ips']
    if webrtc_ips['local_ips']:
        result['leak_detected'] = True
        result['risk_level'] = 'high'
        result['recommendations'].append("Local IP addresses are accessible - WebRTC leak detected")
    
    # Check public IP exposure
    result['public_ips_exposed'] = webrtc_ips['public_ips']
    
    # Test STUN server accessibility
    stun_servers = [
        ("stun.l.google.com", 19302),
        ("stun1.l.google.com", 19302),
        ("stun.stunprotocol.org", 3478),
        ("stun.voiparound.com", 3478)
    ]
    
    for server, port in stun_servers:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(5)
            sock.connect((server, port))
            sock.close()
            result['stun_servers_accessible'].append(f"{server}:{port}")
        except Exception:
            continue
    
    # Simulate ICE candidate gathering
    ice_candidates = simulate_ice_candidate_gathering()
    result['ice_candidates'] = ice_candidates
    
    # Analyze risk level
    if len(result['local_ips_exposed']) > 0:
        result['risk_level'] = 'high'
    elif len(result['stun_servers_accessible']) > 2:
        result['risk_level'] = 'medium'
    
    # Generate recommendations
    if result['leak_detected']:
        result['recommendations'].extend([
            "Disable WebRTC in your browser",
            "Use a VPN with WebRTC leak protection",
            "Configure browser to use proxy for WebRTC",
            "Use browser extensions to block WebRTC"
        ])
    else:
        result['recommendations'].append("No WebRTC leaks detected")
    
    return result


def simulate_ice_candidate_gathering() -> List[Dict]:
    """
    Simulate ICE candidate gathering process.
    
    Returns:
        list: ICE candidate information
    """
    candidates = []
    
    # Get local IPs that could be used as ICE candidates
    webrtc_ips = get_webrtc_ips()
    
    for local_ip in webrtc_ips['local_ips']:
        # Host candidate
        candidates.append({
            'type': 'host',
            'protocol': 'udp',
            'ip': local_ip,
            'port': 50000,  # Example port
            'priority': 2130706431,  # High priority for host candidates
            'foundation': '1'
        })
        
        # Server reflexive candidate (simulated)
        candidates.append({
            'type': 'srflx',
            'protocol': 'udp', 
            'ip': local_ip,
            'port': 50001,
            'priority': 1694498815,
            'foundation': '2',
            'related_address': local_ip,
            'related_port': 50000
        })
    
    return candidates


def test_stun_server(server: str, port: int = 3478) -> Dict:
    """
    Test connectivity to a STUN server.
    
    Args:
        server (str): STUN server hostname
        port (int): STUN server port
        
    Returns:
        dict: STUN server test results
    """
    result = {
        'server': f"{server}:{port}",
        'reachable': False,
        'response_time': None,
        'error': None,
        'mapped_address': None
    }
    
    try:
        start_time = time.time()
        
        # Create STUN binding request
        stun_request = create_stun_binding_request()
        
        # Send request to STUN server
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(10)
        
        sock.sendto(stun_request, (server, port))
        response, addr = sock.recvfrom(1024)
        
        end_time = time.time()
        result['response_time'] = round((end_time - start_time) * 1000, 2)
        
        # Parse STUN response
        mapped_address = parse_stun_response(response)
        if mapped_address:
            result['mapped_address'] = mapped_address
        
        result['reachable'] = True
        sock.close()
    
    except socket.timeout:
        result['error'] = "Connection timeout"
    except Exception as e:
        result['error'] = str(e)
    
    return result


def create_stun_binding_request() -> bytes:
    """
    Create a STUN binding request message.
    
    Returns:
        bytes: STUN binding request
    """
    # STUN message format:
    # 0                   1                   2                   3
    # 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # |0 0|     STUN Message Type     |         Message Length        |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # |                         Magic Cookie                          |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # |                     Transaction ID (96 bits)                  |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    
    import struct
    import random
    
    # STUN binding request (0x0001)
    message_type = 0x0001
    message_length = 0
    magic_cookie = 0x2112A442
    transaction_id = random.getrandbits(96)
    
    return struct.pack('!HHIQ', message_type, message_length, magic_cookie, transaction_id)


def parse_stun_response(response: bytes) -> Optional[str]:
    """
    Parse STUN response to extract mapped address.
    
    Args:
        response (bytes): STUN response message
        
    Returns:
        str: Mapped IP address if found, None otherwise
    """
    try:
        import struct
        
        if len(response) < 20:
            return None
        
        # Parse STUN header
        message_type, message_length, magic_cookie, transaction_id = struct.unpack('!HHIQ', response[:20])
        
        # Check if it's a binding response (0x0101)
        if message_type != 0x0101:
            return None
        
        # Parse attributes
        offset = 20
        while offset < len(response) - 4:
            attr_type, attr_length = struct.unpack('!HH', response[offset:offset+4])
            offset += 4
            
            if attr_type == 0x0001:  # MAPPED-ADDRESS
                if attr_length >= 8:
                    family, port, ip_bytes = struct.unpack('!HH4s', response[offset:offset+8])
                    if family == 0x0001:  # IPv4
                        ip = socket.inet_ntoa(ip_bytes)
                        return ip
            elif attr_type == 0x0020:  # XOR-MAPPED-ADDRESS
                if attr_length >= 8:
                    family, port, ip_bytes = struct.unpack('!HH4s', response[offset:offset+8])
                    if family == 0x0001:  # IPv4
                        # XOR with magic cookie
                        magic_cookie_bytes = struct.pack('!I', 0x2112A442)
                        ip_bytes = bytes(a ^ b for a, b in zip(ip_bytes, magic_cookie_bytes))
                        ip = socket.inet_ntoa(ip_bytes)
                        return ip
            
            # Move to next attribute
            offset += attr_length
            # Align to 4-byte boundary
            offset = (offset + 3) & ~3
    
    except Exception:
        pass
    
    return None


def get_webrtc_fingerprint() -> Dict:
    """
    Get WebRTC fingerprint information.
    
    Returns:
        dict: WebRTC fingerprint data
    """
    fingerprint = {
        'user_agent': None,
        'platform': platform.system(),
        'local_ips': [],
        'public_ips': [],
        'timezone': None,
        'language': None,
        'screen_resolution': None,
        'webrtc_support': False
    }
    
    # Get IP information
    webrtc_ips = get_webrtc_ips()
    fingerprint['local_ips'] = webrtc_ips['local_ips']
    fingerprint['public_ips'] = webrtc_ips['public_ips']
    
    # Get timezone
    try:
        import datetime
        fingerprint['timezone'] = str(datetime.datetime.now().astimezone().tzinfo)
    except Exception:
        pass
    
    # Get language
    try:
        import locale
        fingerprint['language'] = locale.getdefaultlocale()[0]
    except Exception:
        pass
    
    # Check WebRTC support (simplified)
    fingerprint['webrtc_support'] = True  # Assume supported for this implementation
    
    return fingerprint


def generate_webrtc_test_html() -> str:
    """
    Generate HTML for WebRTC leak testing.
    
    Returns:
        str: HTML content for WebRTC testing
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WebRTC Leak Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .result { margin: 10px 0; padding: 10px; border: 1px solid #ccc; }
            .leak { background-color: #ffebee; border-color: #f44336; }
            .safe { background-color: #e8f5e8; border-color: #4caf50; }
            button { padding: 10px 20px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>WebRTC Leak Test</h1>
        <button onclick="testWebRTC()">Test WebRTC</button>
        <div id="results"></div>
        
        <script>
        function testWebRTC() {
            const results = document.getElementById('results');
            results.innerHTML = '<p>Testing WebRTC...</p>';
            
            const pc = new RTCPeerConnection({
                iceServers: [
                    { urls: 'stun:stun.l.google.com:19302' },
                    { urls: 'stun:stun1.l.google.com:19302' }
                ]
            });
            
            const localIPs = [];
            const publicIPs = [];
            
            pc.onicecandidate = function(event) {
                if (event.candidate) {
                    const candidate = event.candidate.candidate;
                    console.log('ICE Candidate:', candidate);
                    
                    // Parse candidate
                    const parts = candidate.split(' ');
                    if (parts.length >= 4) {
                        const type = parts[7];
                        const ip = parts[4];
                        
                        if (type === 'host') {
                            localIPs.push(ip);
                        } else if (type === 'srflx') {
                            publicIPs.push(ip);
                        }
                    }
                }
            };
            
            // Create data channel to trigger ICE gathering
            pc.createDataChannel('test');
            pc.createOffer().then(offer => {
                pc.setLocalDescription(offer);
            });
            
            // Wait for ICE gathering to complete
            setTimeout(() => {
                let html = '<h2>Test Results</h2>';
                
                if (localIPs.length > 0) {
                    html += '<div class="result leak">';
                    html += '<h3>⚠️ WebRTC Leak Detected!</h3>';
                    html += '<p>Local IPs exposed: ' + localIPs.join(', ') + '</p>';
                    html += '</div>';
                } else {
                    html += '<div class="result safe">';
                    html += '<h3>✅ No WebRTC Leak</h3>';
                    html += '<p>No local IPs were exposed</p>';
                    html += '</div>';
                }
                
                if (publicIPs.length > 0) {
                    html += '<div class="result">';
                    html += '<h3>Public IPs</h3>';
                    html += '<p>Public IPs: ' + publicIPs.join(', ') + '</p>';
                    html += '</div>';
                }
                
                results.innerHTML = html;
            }, 3000);
        }
        </script>
    </body>
    </html>
    """
    
    return html


def save_webrtc_test_html(filename: str = "webrtc_test.html") -> str:
    """
    Save WebRTC test HTML to a file.
    
    Args:
        filename (str): Output filename
        
    Returns:
        str: Path to saved file
    """
    html_content = generate_webrtc_test_html()
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return filename
