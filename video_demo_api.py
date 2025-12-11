"""
API Demo Script for Video Recording
Demonstrates API calls matching VIDEO_SCRIPT_GUIDE.md
"""

import requests
import json
import time

# API endpoint (make sure API server is running)
API_URL = "http://localhost:5000"

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def demo_api_calls():
    """Demo API calls for video recording"""
    
    print_section("🎬 IBM WATSON AI - API DEMO FOR VIDEO")
    
    # Check if API is running
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        print("\n✅ API Server is running")
        print(f"   Status: {response.json()['status']}")
    except requests.exceptions.RequestException:
        print("\n❌ ERROR: API server is not running!")
        print("   Please start the API server first:")
        print("   python src/api_service.py")
        return
    
    # Scene 3: Normal Transaction
    print_section("🎬 SCENE 3: NORMAL TRANSACTION")
    
    print("\n📝 Request:")
    normal_txn = {
        "transaction_id": "TXN_001",
        "user_id": "DEMO_USER",
        "amount": 49.99,
        "merchant_category": "grocery",
        "location": "US",
        "device_id": "DEVICE_A"
    }
    print(json.dumps(normal_txn, indent=2))
    
    print("\n⏱️  Making API call to POST /api/v1/predict...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_URL}/api/v1/predict",
            json=normal_txn,
            headers={"Content-Type": "application/json"}
        )
        latency_ms = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ Response:")
            display_result = {
                "risk_score": result.get('risk_score', 0),
                "decision": result.get('decision', 'unknown'),
                "action": result.get('action', 'none'),
                "confidence": result.get('confidence', 'unknown'),
                "latency_ms": round(latency_ms, 2)
            }
            print(json.dumps(display_result, indent=2))
            
            print("\n💬 Narration:")
            print(f"   Risk score: {result.get('risk_score', 0)} out of 100.")
            print(f"   Decision: {result.get('decision', 'unknown').upper()}.")
            print("   No customer friction.")
            print(f"   Transaction completed in {latency_ms:.0f} milliseconds.")
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    # Wait a moment between demos
    time.sleep(2)
    
    # Scene 4: Fraudulent Transaction
    print_section("🎬 SCENE 4: FRAUDULENT TRANSACTION")
    
    print("\n📝 Request:")
    fraud_txn = {
        "transaction_id": "TXN_002",
        "user_id": "DEMO_USER",
        "amount": 5000.00,
        "merchant_category": "gambling",
        "location": "CN",
        "device_id": "UNKNOWN"
    }
    print(json.dumps(fraud_txn, indent=2))
    
    print("\n⏱️  Making API call to POST /api/v1/predict...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_URL}/api/v1/predict",
            json=fraud_txn,
            headers={"Content-Type": "application/json"}
        )
        latency_ms = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n🚨 Response:")
            display_result = {
                "risk_score": result.get('risk_score', 0),
                "decision": result.get('decision', 'unknown'),
                "action": result.get('action', 'none'),
                "top_risk_factors": result.get('top_risk_factors', [])[:4]
            }
            print(json.dumps(display_result, indent=2))
            
            print("\n💬 Narration:")
            print(f"   Risk score: {result.get('risk_score', 0)}.")
            print(f"   Decision: {result.get('decision', 'unknown').upper()}.")
            print("   The system immediately identifies this as fraud and provides")
            print("   explainable reasons:")
            
            for factor in result.get('top_risk_factors', [])[:4]:
                factor_name = factor.get('factor', 'unknown').replace('_', ' ').title()
                contribution = factor.get('contribution', 0)
                print(f"     • {factor_name} (contribution: {contribution})")
            
            print("\n   The fraud team gets an instant alert with all the context they need.")
            print("   No guesswork, just intelligent AI-powered decisions.")
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    # Summary
    print_section("✅ API DEMO COMPLETED")
    print("\n📹 Ready for screen recording!")
    print("   These API calls match the video script guide.")
    print("="*80)


def demo_curl_commands():
    """Show equivalent curl commands for the demo"""
    
    print_section("📋 EQUIVALENT CURL COMMANDS")
    
    print("\n🎬 Scene 3: Normal Transaction")
    print("\ncurl -X POST http://localhost:5000/api/v1/predict \\")
    print('  -H "Content-Type: application/json" \\')
    print("  -d '{")
    print('    "transaction_id": "TXN_001",')
    print('    "user_id": "DEMO_USER",')
    print('    "amount": 49.99,')
    print('    "merchant_category": "grocery",')
    print('    "location": "US",')
    print('    "device_id": "DEVICE_A"')
    print("  }'")
    
    print("\n🎬 Scene 4: Fraudulent Transaction")
    print("\ncurl -X POST http://localhost:5000/api/v1/predict \\")
    print('  -H "Content-Type: application/json" \\')
    print("  -d '{")
    print('    "transaction_id": "TXN_002",')
    print('    "user_id": "DEMO_USER",')
    print('    "amount": 5000.00,')
    print('    "merchant_category": "gambling",')
    print('    "location": "CN",')
    print('    "device_id": "UNKNOWN"')
    print("  }'")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--curl":
        # Show curl commands
        demo_curl_commands()
    else:
        # Run API demo
        print("\n⚠️  IMPORTANT: Make sure the API server is running!")
        print("   In another terminal, run: python src/api_service.py")
        print("   Then press Enter to continue...")
        input()
        
        demo_api_calls()
