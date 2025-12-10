"""
Test Script - Quick API Testing
Run this to test the fraud detection API
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:5000"


def test_health():
    """Test health endpoint"""
    print("\n" + "="*70)
    print("Testing Health Endpoint")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_normal_transaction():
    """Test normal transaction"""
    print("\n" + "="*70)
    print("Testing Normal Transaction")
    print("="*70)
    
    transaction = {
        "transaction_id": "TXN001",
        "user_id": "USER123",
        "amount": 45.99,
        "merchant_category": "grocery",
        "location": "US",
        "device_id": "DEVICE_A",
        "latitude": 40.7128,
        "longitude": -74.0060
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/predict",
        json=transaction
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Risk Score: {result.get('risk_score')}")
    print(f"Decision: {result.get('decision')}")
    print(f"Response: {json.dumps(result, indent=2)}")


def test_fraudulent_transaction():
    """Test potentially fraudulent transaction"""
    print("\n" + "="*70)
    print("Testing Fraudulent Transaction")
    print("="*70)
    
    transaction = {
        "transaction_id": "TXN002",
        "user_id": "USER123",
        "amount": 5000.00,
        "merchant_category": "crypto",
        "location": "RU",
        "device_id": "DEVICE_NEW",
        "latitude": 55.7558,
        "longitude": 37.6173
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/predict",
        json=transaction
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Risk Score: {result.get('risk_score')}")
    print(f"Decision: {result.get('decision')}")
    print(f"Response: {json.dumps(result, indent=2)}")


def test_batch_prediction():
    """Test batch prediction"""
    print("\n" + "="*70)
    print("Testing Batch Prediction")
    print("="*70)
    
    transactions = {
        "transactions": [
            {
                "transaction_id": "TXN003",
                "user_id": "USER456",
                "amount": 89.50,
                "merchant_category": "restaurant",
                "location": "US"
            },
            {
                "transaction_id": "TXN004",
                "user_id": "USER456",
                "amount": 2500.00,
                "merchant_category": "gambling",
                "location": "CN"
            }
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/predict/batch",
        json=transactions
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Total Transactions: {result.get('total_transactions')}")
    for i, txn_result in enumerate(result.get('results', []), 1):
        print(f"\nTransaction {i}:")
        print(f"  ID: {txn_result.get('transaction_id')}")
        print(f"  Risk Score: {txn_result.get('risk_score')}")
        print(f"  Decision: {txn_result.get('decision')}")


def test_monitoring():
    """Test real-time monitoring"""
    print("\n" + "="*70)
    print("Testing Real-Time Monitoring")
    print("="*70)
    
    transaction = {
        "transaction_id": "TXN005",
        "user_id": "USER789",
        "amount": 150.00,
        "merchant_category": "online",
        "location": "US",
        "device_id": "DEVICE_X",
        "latitude": 34.0522,
        "longitude": -118.2437
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/monitor",
        json=transaction
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Risk Score: {result.get('risk_score')}")
    print(f"Decision: {result.get('decision')}")
    print(f"Processing Time: {result.get('processing_time_ms'):.2f}ms")
    print(f"Flags: {len(result.get('flags', []))}")


def test_user_stats():
    """Test user statistics"""
    print("\n" + "="*70)
    print("Testing User Statistics")
    print("="*70)
    
    user_id = "USER123"
    response = requests.get(f"{BASE_URL}/api/v1/stats/user/{user_id}")
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
    else:
        print(f"Response: {response.json()}")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("FRAUD DETECTION API TEST SUITE")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        test_health()
        test_normal_transaction()
        test_fraudulent_transaction()
        test_batch_prediction()
        test_monitoring()
        test_user_stats()
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API server")
        print("Make sure the API is running: python src/api_service.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


if __name__ == "__main__":
    run_all_tests()
