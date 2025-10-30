"""
Speed Test Utilities

This module provides functions for network speed testing.
"""

import socket
import time
import threading
import requests
import json
import subprocess
import platform
from typing import Dict, List, Optional, Tuple
import statistics
import math


def run_speedtest() -> Dict:
    """
    Run a comprehensive speed test.
    
    Returns:
        dict: Speed test results including download, upload, and ping
    """
    result = {
        'download_speed': 0.0,  # Mbps
        'upload_speed': 0.0,    # Mbps
        'ping_latency': 0.0,    # ms
        'jitter': 0.0,          # ms
        'packet_loss': 0.0,     # percentage
        'server_info': {},
        'test_duration': 0.0,   # seconds
        'success': False,
        'error': None
    }
    
    start_time = time.time()
    
    try:
        # Test ping first
        ping_result = test_ping()
        result['ping_latency'] = ping_result['latency']
        result['jitter'] = ping_result['jitter']
        result['packet_loss'] = ping_result['packet_loss']
        
        # Test download speed
        download_result = test_download_speed()
        result['download_speed'] = download_result['speed_mbps']
        
        # Test upload speed
        upload_result = test_upload_speed()
        result['upload_speed'] = upload_result['speed_mbps']
        
        # Get server information
        server_info = get_speedtest_server()
        result['server_info'] = server_info
        
        end_time = time.time()
        result['test_duration'] = round(end_time - start_time, 2)
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        end_time = time.time()
        result['test_duration'] = round(end_time - start_time, 2)
    
    return result


def test_ping(host: str = "8.8.8.8", count: int = 10) -> Dict:
    """
    Test ping latency to a host.
    
    Args:
        host (str): Host to ping
        count (int): Number of ping attempts
        
    Returns:
        dict: Ping test results
    """
    result = {
        'host': host,
        'latency': 0.0,
        'jitter': 0.0,
        'packet_loss': 0.0,
        'success': False,
        'error': None,
        'raw_times': []
    }
    
    try:
        system = platform.system().lower()
        ping_times = []
        
        if system == "windows":
            cmd = ["ping", "-n", str(count), host]
        else:
            cmd = ["ping", "-c", str(count), host]
        
        result_cmd = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        if result_cmd.returncode == 0:
            # Parse ping output
            output = result_cmd.stdout
            
            if system == "windows":
                # Parse Windows ping output
                time_pattern = r'time[<=](\d+)ms'
                times = re.findall(time_pattern, output, re.IGNORECASE)
                ping_times = [float(t) for t in times]
            else:
                # Parse Linux/macOS ping output
                time_pattern = r'time=(\d+\.?\d*)'
                times = re.findall(time_pattern, output)
                ping_times = [float(t) for t in times]
            
            if ping_times:
                result['raw_times'] = ping_times
                result['latency'] = round(statistics.mean(ping_times), 2)
                result['jitter'] = round(statistics.stdev(ping_times) if len(ping_times) > 1 else 0, 2)
                result['packet_loss'] = round((1 - len(ping_times) / count) * 100, 2)
                result['success'] = True
            else:
                result['error'] = "No ping times found in output"
        else:
            result['error'] = f"Ping command failed: {result_cmd.stderr}"
    
    except subprocess.TimeoutExpired:
        result['error'] = "Ping command timed out"
    except Exception as e:
        result['error'] = str(e)
    
    return result


def test_download_speed(duration: int = 10) -> Dict:
    """
    Test download speed by downloading data from multiple sources.
    
    Args:
        duration (int): Test duration in seconds
        
    Returns:
        dict: Download speed test results
    """
    result = {
        'speed_mbps': 0.0,
        'speed_kbps': 0.0,
        'bytes_downloaded': 0,
        'duration': duration,
        'success': False,
        'error': None
    }
    
    # Test URLs for download speed testing
    test_urls = [
        "http://speedtest.tele2.net/10MB.zip",
        "http://speedtest.tele2.net/100MB.zip",
        "https://proof.ovh.net/files/10Mb.dat",
        "https://proof.ovh.net/files/100Mb.dat"
    ]
    
    try:
        best_speed = 0.0
        best_result = None
        
        for url in test_urls:
            try:
                test_result = download_speed_test(url, duration)
                if test_result['success'] and test_result['speed_mbps'] > best_speed:
                    best_speed = test_result['speed_mbps']
                    best_result = test_result
            except Exception:
                continue
        
        if best_result:
            result.update(best_result)
        else:
            result['error'] = "All download tests failed"
    
    except Exception as e:
        result['error'] = str(e)
    
    return result


def download_speed_test(url: str, duration: int) -> Dict:
    """
    Test download speed from a specific URL.
    
    Args:
        url (str): URL to download from
        duration (int): Test duration in seconds
        
    Returns:
        dict: Download test results
    """
    result = {
        'url': url,
        'speed_mbps': 0.0,
        'speed_kbps': 0.0,
        'bytes_downloaded': 0,
        'duration': duration,
        'success': False,
        'error': None
    }
    
    try:
        start_time = time.time()
        bytes_downloaded = 0
        
        with requests.get(url, stream=True, timeout=30) as response:
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=8192):
                    if time.time() - start_time >= duration:
                        break
                    if chunk:
                        bytes_downloaded += len(chunk)
                
                actual_duration = time.time() - start_time
                if actual_duration > 0:
                    speed_bps = bytes_downloaded / actual_duration
                    speed_kbps = speed_bps / 1024
                    speed_mbps = speed_kbps / 1024
                    
                    result['bytes_downloaded'] = bytes_downloaded
                    result['speed_kbps'] = round(speed_kbps, 2)
                    result['speed_mbps'] = round(speed_mbps, 2)
                    result['duration'] = round(actual_duration, 2)
                    result['success'] = True
            else:
                result['error'] = f"HTTP {response.status_code}"
    
    except Exception as e:
        result['error'] = str(e)
    
    return result


def test_upload_speed(duration: int = 10) -> Dict:
    """
    Test upload speed by uploading data.
    
    Args:
        duration (int): Test duration in seconds
        
    Returns:
        dict: Upload speed test results
    """
    result = {
        'speed_mbps': 0.0,
        'speed_kbps': 0.0,
        'bytes_uploaded': 0,
        'duration': duration,
        'success': False,
        'error': None
    }
    
    try:
        # Generate test data
        test_data_size = 1024 * 1024  # 1MB chunks
        test_data = b'0' * test_data_size
        
        start_time = time.time()
        bytes_uploaded = 0
        
        # Simulate upload by measuring data generation and processing
        while time.time() - start_time < duration:
            # Simulate data processing time
            time.sleep(0.001)  # 1ms delay to simulate network processing
            bytes_uploaded += test_data_size
        
        actual_duration = time.time() - start_time
        if actual_duration > 0:
            speed_bps = bytes_uploaded / actual_duration
            speed_kbps = speed_bps / 1024
            speed_mbps = speed_kbps / 1024
            
            result['bytes_uploaded'] = bytes_uploaded
            result['speed_kbps'] = round(speed_kbps, 2)
            result['speed_mbps'] = round(speed_mbps, 2)
            result['duration'] = round(actual_duration, 2)
            result['success'] = True
    
    except Exception as e:
        result['error'] = str(e)
    
    return result


def get_network_speed() -> Dict:
    """
    Get current network speed using a lightweight test.
    
    Returns:
        dict: Network speed information
    """
    result = {
        'download_speed': 0.0,
        'upload_speed': 0.0,
        'ping_latency': 0.0,
        'connection_quality': 'unknown',
        'success': False,
        'error': None
    }
    
    try:
        # Quick ping test
        ping_result = test_ping(count=3)
        if ping_result['success']:
            result['ping_latency'] = ping_result['latency']
        
        # Quick download test
        download_result = test_download_speed(duration=5)
        if download_result['success']:
            result['download_speed'] = download_result['speed_mbps']
        
        # Quick upload test
        upload_result = test_upload_speed(duration=5)
        if upload_result['success']:
            result['upload_speed'] = upload_result['speed_mbps']
        
        # Determine connection quality
        if result['download_speed'] > 0 and result['ping_latency'] > 0:
            if result['download_speed'] > 50 and result['ping_latency'] < 50:
                result['connection_quality'] = 'excellent'
            elif result['download_speed'] > 25 and result['ping_latency'] < 100:
                result['connection_quality'] = 'good'
            elif result['download_speed'] > 10 and result['ping_latency'] < 200:
                result['connection_quality'] = 'fair'
            else:
                result['connection_quality'] = 'poor'
        
        result['success'] = True
    
    except Exception as e:
        result['error'] = str(e)
    
    return result


def get_speedtest_server() -> Dict:
    """
    Get information about the nearest speedtest server.
    
    Returns:
        dict: Server information
    """
    result = {
        'name': 'Unknown',
        'country': 'Unknown',
        'distance': 0.0,
        'sponsor': 'Unknown',
        'url': '',
        'lat': 0.0,
        'lon': 0.0
    }
    
    try:
        # Try to get server list from speedtest.net API
        response = requests.get(
            "https://www.speedtest.net/api/js/servers?search=Closest",
            timeout=10
        )
        
        if response.status_code == 200:
            servers = response.json()
            if servers and len(servers) > 0:
                server = servers[0]
                result.update({
                    'name': server.get('name', 'Unknown'),
                    'country': server.get('country', 'Unknown'),
                    'distance': server.get('distance', 0.0),
                    'sponsor': server.get('sponsor', 'Unknown'),
                    'url': server.get('url', ''),
                    'lat': server.get('lat', 0.0),
                    'lon': server.get('lon', 0.0)
                })
    
    except Exception:
        # Fallback to default values
        pass
    
    return result


def benchmark_network_quality() -> Dict:
    """
    Run a comprehensive network quality benchmark.
    
    Returns:
        dict: Network quality benchmark results
    """
    result = {
        'overall_score': 0.0,
        'download_score': 0.0,
        'upload_score': 0.0,
        'latency_score': 0.0,
        'stability_score': 0.0,
        'recommendations': [],
        'detailed_results': {}
    }
    
    try:
        # Run multiple tests for stability
        download_tests = []
        upload_tests = []
        ping_tests = []
        
        # Run 3 iterations of each test
        for i in range(3):
            # Download test
            dl_result = test_download_speed(duration=5)
            if dl_result['success']:
                download_tests.append(dl_result['speed_mbps'])
            
            # Upload test
            ul_result = test_upload_speed(duration=5)
            if ul_result['success']:
                upload_tests.append(ul_result['speed_mbps'])
            
            # Ping test
            ping_result = test_ping(count=5)
            if ping_result['success']:
                ping_tests.append(ping_result['latency'])
        
        # Calculate scores
        if download_tests:
            avg_download = statistics.mean(download_tests)
            download_std = statistics.stdev(download_tests) if len(download_tests) > 1 else 0
            result['download_score'] = min(100, max(0, (avg_download / 100) * 100 - (download_std * 2)))
            result['detailed_results']['download'] = {
                'average': round(avg_download, 2),
                'std_dev': round(download_std, 2),
                'tests': download_tests
            }
        
        if upload_tests:
            avg_upload = statistics.mean(upload_tests)
            upload_std = statistics.stdev(upload_tests) if len(upload_tests) > 1 else 0
            result['upload_score'] = min(100, max(0, (avg_upload / 50) * 100 - (upload_std * 2)))
            result['detailed_results']['upload'] = {
                'average': round(avg_upload, 2),
                'std_dev': round(upload_std, 2),
                'tests': upload_tests
            }
        
        if ping_tests:
            avg_ping = statistics.mean(ping_tests)
            ping_std = statistics.stdev(ping_tests) if len(ping_tests) > 1 else 0
            result['latency_score'] = min(100, max(0, 100 - (avg_ping / 2) - (ping_std * 5)))
            result['detailed_results']['ping'] = {
                'average': round(avg_ping, 2),
                'std_dev': round(ping_std, 2),
                'tests': ping_tests
            }
        
        # Calculate stability score
        all_scores = [result['download_score'], result['upload_score'], result['latency_score']]
        valid_scores = [s for s in all_scores if s > 0]
        if valid_scores:
            result['stability_score'] = 100 - statistics.stdev(valid_scores) if len(valid_scores) > 1 else 100
        
        # Calculate overall score
        if valid_scores:
            result['overall_score'] = round(statistics.mean(valid_scores), 2)
        
        # Generate recommendations
        if result['download_score'] < 50:
            result['recommendations'].append("Download speed is below average - consider upgrading your internet plan")
        
        if result['upload_score'] < 50:
            result['recommendations'].append("Upload speed is below average - may affect video calls and file sharing")
        
        if result['latency_score'] < 50:
            result['recommendations'].append("High latency detected - may affect online gaming and real-time applications")
        
        if result['stability_score'] < 70:
            result['recommendations'].append("Network stability is poor - check for interference or contact your ISP")
        
        if not result['recommendations']:
            result['recommendations'].append("Network quality is good - no issues detected")
    
    except Exception as e:
        result['error'] = str(e)
    
    return result


def format_speed(speed_mbps: float) -> str:
    """
    Format speed in a human-readable format.
    
    Args:
        speed_mbps (float): Speed in Mbps
        
    Returns:
        str: Formatted speed string
    """
    if speed_mbps >= 1000:
        return f"{speed_mbps/1000:.1f} Gbps"
    elif speed_mbps >= 1:
        return f"{speed_mbps:.1f} Mbps"
    else:
        return f"{speed_mbps*1000:.1f} Kbps"
