"""
Utility functions for Financial Fraud Detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import hashlib

logger = logging.getLogger(__name__)


def format_currency(amount, currency='USD'):
    """Format amount as currency"""
    symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥'
    }
    symbol = symbols.get(currency, '$')
    return f"{symbol}{amount:,.2f}"


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth
    Returns distance in kilometers
    """
    R = 6371  # Earth radius in kilometers
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def generate_device_fingerprint(user_agent, ip_address, screen_resolution=None):
    """Generate a device fingerprint"""
    fingerprint_data = f"{user_agent}|{ip_address}"
    if screen_resolution:
        fingerprint_data += f"|{screen_resolution}"
    
    return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]


def is_business_hours(timestamp, timezone='UTC'):
    """Check if timestamp is during business hours (9 AM - 5 PM)"""
    hour = timestamp.hour
    return 9 <= hour <= 17


def is_weekend(timestamp):
    """Check if timestamp is on weekend"""
    return timestamp.weekday() >= 5


def calculate_time_difference(timestamp1, timestamp2, unit='hours'):
    """Calculate time difference between two timestamps"""
    diff = abs(timestamp2 - timestamp1)
    
    if unit == 'seconds':
        return diff.total_seconds()
    elif unit == 'minutes':
        return diff.total_seconds() / 60
    elif unit == 'hours':
        return diff.total_seconds() / 3600
    elif unit == 'days':
        return diff.days
    else:
        return diff


def normalize_merchant_category(category):
    """Normalize merchant category names"""
    category = str(category).lower().strip()
    
    # Map common variations to standard categories
    mapping = {
        'grocery': ['grocery', 'supermarket', 'food store'],
        'restaurant': ['restaurant', 'dining', 'food service'],
        'retail': ['retail', 'shopping', 'store'],
        'gas': ['gas', 'fuel', 'petrol'],
        'online': ['online', 'e-commerce', 'internet'],
        'gambling': ['gambling', 'casino', 'betting'],
        'crypto': ['crypto', 'cryptocurrency', 'bitcoin']
    }
    
    for standard, variations in mapping.items():
        if any(var in category for var in variations):
            return standard
    
    return category


def validate_transaction_data(transaction):
    """Validate transaction data structure"""
    required_fields = ['transaction_id', 'user_id', 'amount']
    optional_fields = ['merchant_category', 'location', 'device_id', 
                      'latitude', 'longitude', 'timestamp']
    
    errors = []
    
    # Check required fields
    for field in required_fields:
        if field not in transaction:
            errors.append(f"Missing required field: {field}")
    
    # Validate amount
    if 'amount' in transaction:
        try:
            amount = float(transaction['amount'])
            if amount <= 0:
                errors.append("Amount must be positive")
            if amount > 1000000:
                errors.append("Amount exceeds maximum allowed")
        except (ValueError, TypeError):
            errors.append("Invalid amount format")
    
    # Validate coordinates
    if 'latitude' in transaction or 'longitude' in transaction:
        if 'latitude' in transaction and 'longitude' in transaction:
            try:
                lat = float(transaction['latitude'])
                lon = float(transaction['longitude'])
                if not (-90 <= lat <= 90):
                    errors.append("Invalid latitude")
                if not (-180 <= lon <= 180):
                    errors.append("Invalid longitude")
            except (ValueError, TypeError):
                errors.append("Invalid coordinate format")
        else:
            errors.append("Both latitude and longitude required")
    
    return len(errors) == 0, errors


def create_transaction_summary(transaction, result):
    """Create a summary of transaction and fraud detection result"""
    summary = {
        'transaction': {
            'id': transaction.get('transaction_id'),
            'user_id': transaction.get('user_id'),
            'amount': format_currency(transaction.get('amount', 0)),
            'merchant': transaction.get('merchant_category', 'unknown'),
            'location': transaction.get('location', 'unknown'),
            'timestamp': transaction.get('timestamp', datetime.now()).isoformat()
        },
        'fraud_detection': {
            'risk_score': result.get('risk_score'),
            'decision': result.get('decision'),
            'action': result.get('action'),
            'confidence': result.get('confidence'),
            'top_factors': result.get('top_risk_factors', [])[:3]
        }
    }
    return summary


def export_results_to_csv(results, filepath='fraud_detection_results.csv'):
    """Export fraud detection results to CSV"""
    if not results:
        logger.warning("No results to export")
        return
    
    # Flatten results for CSV
    flattened = []
    for result in results:
        flat = {
            'transaction_id': result.get('transaction_id'),
            'risk_score': result.get('risk_score'),
            'decision': result.get('decision'),
            'action': result.get('action'),
            'confidence': result.get('confidence'),
            'timestamp': result.get('timestamp')
        }
        
        # Add component scores
        if 'component_scores' in result:
            for component, score in result['component_scores'].items():
                flat[f'score_{component}'] = score
        
        flattened.append(flat)
    
    df = pd.DataFrame(flattened)
    df.to_csv(filepath, index=False)
    logger.info(f"Results exported to {filepath}")


def calculate_performance_metrics(predictions, actuals):
    """Calculate performance metrics for fraud detection"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix
    )
    
    metrics = {
        'accuracy': accuracy_score(actuals, predictions),
        'precision': precision_score(actuals, predictions),
        'recall': recall_score(actuals, predictions),
        'f1_score': f1_score(actuals, predictions),
        'confusion_matrix': confusion_matrix(actuals, predictions).tolist()
    }
    
    # Calculate AUC if probabilities available
    if hasattr(predictions, 'predict_proba'):
        probabilities = predictions.predict_proba(actuals)[:, 1]
        metrics['auc_roc'] = roc_auc_score(actuals, probabilities)
    
    return metrics


def generate_alert_message(result):
    """Generate alert message for high-risk transactions"""
    risk_score = result.get('risk_score', 0)
    decision = result.get('decision', 'unknown')
    transaction_id = result.get('transaction_id', 'N/A')
    
    if risk_score >= 70:
        severity = 'CRITICAL'
        emoji = '🚨'
    elif risk_score >= 40:
        severity = 'WARNING'
        emoji = '⚠️'
    else:
        severity = 'INFO'
        emoji = 'ℹ️'
    
    top_factors = result.get('top_risk_factors', [])
    factor_text = ', '.join([f['factor'] for f in top_factors[:3]])
    
    message = f"""
{emoji} {severity}: Fraud Alert
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Transaction ID: {transaction_id}
Risk Score: {risk_score}/100
Decision: {decision.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Top Risk Factors:
{factor_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Action Required: {result.get('action', 'Review transaction')}
"""
    return message.strip()


def load_json_config(filepath):
    """Load configuration from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config from {filepath}: {e}")
        return {}


def save_json_config(config, filepath):
    """Save configuration to JSON file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving config to {filepath}: {e}")


def create_sample_request():
    """Create a sample API request for testing"""
    return {
        "transaction_id": f"TXN{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "user_id": "USER12345",
        "amount": 150.00,
        "merchant_category": "online",
        "location": "US",
        "device_id": "DEVICE_ABC123",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test currency formatting
    print(f"\nCurrency: {format_currency(1234.56)}")
    
    # Test distance calculation
    dist = calculate_haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)
    print(f"Distance NYC to London: {dist:.0f} km")
    
    # Test validation
    sample_txn = create_sample_request()
    valid, errors = validate_transaction_data(sample_txn)
    print(f"\nValidation: {valid}")
    if errors:
        print(f"Errors: {errors}")
    
    print("\nUtility tests completed!")
