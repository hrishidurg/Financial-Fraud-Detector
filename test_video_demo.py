"""
Quick Test Script - Verify Video Demo Setup
Run this before recording your video
"""

import sys
import os

def print_status(message, status):
    """Print status with emoji"""
    emoji = "✅" if status else "❌"
    print(f"{emoji} {message}")

def test_imports():
    """Test if all required modules can be imported"""
    print("\n" + "="*60)
    print("Testing Python Dependencies")
    print("="*60)
    
    modules = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('flask', 'Flask'),
        ('datetime', 'datetime (built-in)'),
    ]
    
    all_ok = True
    for module, name in modules:
        try:
            __import__(module)
            print_status(f"{name} installed", True)
        except ImportError:
            print_status(f"{name} NOT installed", False)
            all_ok = False
    
    return all_ok

def test_files():
    """Test if required files exist"""
    print("\n" + "="*60)
    print("Testing Required Files")
    print("="*60)
    
    files = [
        'demo.py',
        'video_demo_api.py',
        'src/api_service.py',
        'src/risk_scoring.py',
        'src/real_time_monitor.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_ok = True
    for file in files:
        exists = os.path.exists(file)
        print_status(f"{file}", exists)
        if not exists:
            all_ok = False
    
    return all_ok

def test_demo_script():
    """Test if demo script runs"""
    print("\n" + "="*60)
    print("Testing Demo Script")
    print("="*60)
    
    try:
        # Try importing the demo modules
        sys.path.append('src')
        from risk_scoring import RiskScoringEngine
        from real_time_monitor import TransactionMonitor
        
        print_status("Demo modules can be imported", True)
        
        # Test basic functionality
        risk_engine = RiskScoringEngine()
        monitor = TransactionMonitor()
        
        print_status("Risk engine initialized", True)
        print_status("Transaction monitor initialized", True)
        
        return True
    except Exception as e:
        print_status(f"Demo script test failed: {e}", False)
        return False

def test_video_demo():
    """Test video demo scenarios"""
    print("\n" + "="*60)
    print("Testing Video Demo Scenarios")
    print("="*60)
    
    try:
        from datetime import datetime
        from risk_scoring import RiskScoringEngine
        from real_time_monitor import TransactionMonitor
        
        risk_engine = RiskScoringEngine()
        monitor = TransactionMonitor()
        
        # Test normal transaction
        normal_txn = {
            "transaction_id": "TEST_001",
            "user_id": "TEST_USER",
            "amount": 49.99,
            "merchant_category": "grocery",
            "location": "US",
            "device_id": "DEVICE_A",
            "timestamp": datetime.now()
        }
        
        monitor.add_transaction(normal_txn)
        result = risk_engine.calculate_risk_score(
            transaction=normal_txn,
            user_history=[],
            user_devices=set()
        )
        
        print_status("Normal transaction test passed", True)
        print(f"   Risk Score: {result['risk_score']}")
        print(f"   Decision: {result['decision']}")
        
        # Test fraud transaction
        fraud_txn = {
            "transaction_id": "TEST_002",
            "user_id": "TEST_USER",
            "amount": 5000.00,
            "merchant_category": "gambling",
            "location": "CN",
            "device_id": "UNKNOWN",
            "timestamp": datetime.now()
        }
        
        result = risk_engine.calculate_risk_score(
            transaction=fraud_txn,
            user_history=[normal_txn],
            user_devices={'DEVICE_A'}
        )
        
        print_status("Fraud transaction test passed", True)
        print(f"   Risk Score: {result['risk_score']}")
        print(f"   Decision: {result['decision']}")
        
        return True
    except Exception as e:
        print_status(f"Video demo test failed: {e}", False)
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🎬 VIDEO DEMO SETUP TEST")
    print("="*60)
    print("\nThis script verifies your setup is ready for video recording.")
    
    results = []
    
    # Run tests
    results.append(("Dependencies", test_imports()))
    results.append(("Required Files", test_files()))
    results.append(("Demo Script", test_demo_script()))
    results.append(("Video Scenarios", test_video_demo()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = all(result[1] for result in results)
    
    for name, passed in results:
        print_status(f"{name}", passed)
    
    print("\n" + "="*60)
    
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\n🎉 Your setup is ready for video recording!")
        print("\nNext steps:")
        print("  1. Run: python demo.py --video")
        print("  2. Start screen recording")
        print("  3. Narrate the output")
        print("  4. Upload to YouTube")
        print("\nSee VIDEO_DEMO_README.md for detailed instructions.")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before recording.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Check you're in the right directory")
        print("  - Verify all files are present")
    
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
