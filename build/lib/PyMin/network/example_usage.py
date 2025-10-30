"""
Example usage of PyMin Network utilities.

This script demonstrates how to use the network utilities for:
- IP address detection
- DNS leak testing
- WebRTC leak testing
- Speed testing
"""

from ip_utils import get_internal_ip, get_external_ip, get_all_ips, get_ip_info
from dns_utils import get_dns_servers, check_dns_leak, test_dns_resolution
from webrtc_utils import check_webrtc_leak, get_webrtc_ips, save_webrtc_test_html
from speedtest_utils import run_speedtest, get_network_speed, benchmark_network_quality


def main():
    """Main function demonstrating network utilities."""
    print("=" * 60)
    print("PyMin Network Utilities - Example Usage")
    print("=" * 60)
    
    # 1. IP Address Detection
    print("\n1. IP Address Detection")
    print("-" * 30)
    
    internal_ip = get_internal_ip()
    print(f"Internal IP: {internal_ip}")
    
    external_ip = get_external_ip()
    print(f"External IP: {external_ip}")
    
    all_ips = get_all_ips()
    print(f"All IPs: {all_ips}")
    
    if external_ip:
        ip_info = get_ip_info(external_ip)
        if ip_info:
            print(f"IP Info: {ip_info.get('city', 'Unknown')}, {ip_info.get('country', 'Unknown')}")
    
    # 2. DNS Configuration and Leak Testing
    print("\n2. DNS Configuration and Leak Testing")
    print("-" * 40)
    
    dns_servers = get_dns_servers()
    print(f"DNS Servers: {dns_servers}")
    
    # Test DNS resolution
    dns_test = test_dns_resolution("google.com")
    print(f"DNS Test for google.com: {dns_test}")
    
    # Check for DNS leaks
    print("\nChecking for DNS leaks...")
    dns_leak_result = check_dns_leak()
    print(f"DNS Leak Detected: {dns_leak_result['leak_detected']}")
    if dns_leak_result['recommendations']:
        print("Recommendations:")
        for rec in dns_leak_result['recommendations']:
            print(f"  - {rec}")
    
    # 3. WebRTC Leak Testing
    print("\n3. WebRTC Leak Testing")
    print("-" * 25)
    
    webrtc_ips = get_webrtc_ips()
    print(f"WebRTC IPs: {webrtc_ips}")
    
    print("\nChecking for WebRTC leaks...")
    webrtc_leak_result = check_webrtc_leak()
    print(f"WebRTC Leak Detected: {webrtc_leak_result['leak_detected']}")
    print(f"Risk Level: {webrtc_leak_result['risk_level']}")
    if webrtc_leak_result['recommendations']:
        print("Recommendations:")
        for rec in webrtc_leak_result['recommendations']:
            print(f"  - {rec}")
    
    # Generate WebRTC test HTML
    html_file = save_webrtc_test_html("webrtc_test.html")
    print(f"WebRTC test HTML saved to: {html_file}")
    
    # 4. Speed Testing
    print("\n4. Speed Testing")
    print("-" * 20)
    
    print("Running quick network speed test...")
    quick_speed = get_network_speed()
    if quick_speed['success']:
        print(f"Download Speed: {quick_speed['download_speed']:.2f} Mbps")
        print(f"Upload Speed: {quick_speed['upload_speed']:.2f} Mbps")
        print(f"Ping Latency: {quick_speed['ping_latency']:.2f} ms")
        print(f"Connection Quality: {quick_speed['connection_quality']}")
    
    print("\nRunning comprehensive speed test...")
    speed_test = run_speedtest()
    if speed_test['success']:
        print(f"Download Speed: {speed_test['download_speed']:.2f} Mbps")
        print(f"Upload Speed: {speed_test['upload_speed']:.2f} Mbps")
        print(f"Ping Latency: {speed_test['ping_latency']:.2f} ms")
        print(f"Jitter: {speed_test['jitter']:.2f} ms")
        print(f"Packet Loss: {speed_test['packet_loss']:.2f}%")
        print(f"Test Duration: {speed_test['test_duration']:.2f} seconds")
    
    # 5. Network Quality Benchmark
    print("\n5. Network Quality Benchmark")
    print("-" * 30)
    
    print("Running network quality benchmark...")
    benchmark = benchmark_network_quality()
    if 'error' not in benchmark:
        print(f"Overall Score: {benchmark['overall_score']:.1f}/100")
        print(f"Download Score: {benchmark['download_score']:.1f}/100")
        print(f"Upload Score: {benchmark['upload_score']:.1f}/100")
        print(f"Latency Score: {benchmark['latency_score']:.1f}/100")
        print(f"Stability Score: {benchmark['stability_score']:.1f}/100")
        
        if benchmark['recommendations']:
            print("\nRecommendations:")
            for rec in benchmark['recommendations']:
                print(f"  - {rec}")
    
    print("\n" + "=" * 60)
    print("Network utilities demonstration completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
