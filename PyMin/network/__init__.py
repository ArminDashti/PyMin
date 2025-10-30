"""
PyMin Network Utilities Module

This module provides network-related utilities including:
- IP address detection (internal and external)
- DNS configuration and leak testing
- WebRTC leak testing
- Speed testing
"""

from .ip_utils import get_internal_ip, get_external_ip, get_all_ips
from .dns_utils import get_dns_servers, check_dns_leak, test_dns_resolution
from .webrtc_utils import check_webrtc_leak, get_webrtc_ips
from .speedtest_utils import run_speedtest, get_network_speed

__all__ = [
    'get_internal_ip',
    'get_external_ip', 
    'get_all_ips',
    'get_dns_servers',
    'check_dns_leak',
    'test_dns_resolution',
    'check_webrtc_leak',
    'get_webrtc_ips',
    'run_speedtest',
    'get_network_speed'
]
