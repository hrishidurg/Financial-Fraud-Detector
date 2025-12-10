"""
Real-Time Transaction Monitoring Module
Monitors transactions in real-time and detects fraud patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionMonitor:
    """Real-time transaction monitoring system"""
    
    def __init__(self, lookback_hours=24):
        self.lookback_hours = lookback_hours
        self.user_transactions = defaultdict(lambda: deque(maxlen=1000))
        self.user_locations = defaultdict(lambda: deque(maxlen=100))
        self.user_devices = defaultdict(set)
        self.merchant_stats = defaultdict(lambda: {'count': 0, 'fraud_count': 0})
        
    def add_transaction(self, transaction):
        """Add transaction to monitoring system"""
        user_id = transaction['user_id']
        timestamp = transaction.get('timestamp', datetime.now())
        
        # Store transaction
        self.user_transactions[user_id].append({
            'timestamp': timestamp,
            'amount': transaction['amount'],
            'location': transaction.get('location'),
            'merchant': transaction.get('merchant_category'),
            'lat': transaction.get('latitude'),
            'lon': transaction.get('longitude')
        })
        
        # Store location
        if 'latitude' in transaction and 'longitude' in transaction:
            self.user_locations[user_id].append({
                'timestamp': timestamp,
                'lat': transaction['latitude'],
                'lon': transaction['longitude']
            })
        
        # Store device
        if 'device_id' in transaction:
            self.user_devices[user_id].add(transaction['device_id'])
        
        # Update merchant stats
        merchant = transaction.get('merchant_category', 'unknown')
        self.merchant_stats[merchant]['count'] += 1
    
    def check_velocity_attack(self, transaction):
        """Check for velocity-based fraud (multiple transactions in short time)"""
        user_id = transaction['user_id']
        current_time = transaction.get('timestamp', datetime.now())
        
        # Get recent transactions
        recent_transactions = [
            t for t in self.user_transactions[user_id]
            if (current_time - t['timestamp']).total_seconds() < 600  # Last 10 minutes
        ]
        
        flags = []
        risk_score = 0
        
        # Check transaction count
        if len(recent_transactions) > 5:
            flags.append({
                'type': 'high_velocity',
                'description': f'{len(recent_transactions)} transactions in 10 minutes',
                'severity': 'high',
                'risk_contribution': 30
            })
            risk_score += 30
        
        # Check for multiple different locations
        unique_locations = len(set(t.get('location') for t in recent_transactions if t.get('location')))
        if unique_locations > 3:
            flags.append({
                'type': 'multiple_locations',
                'description': f'{unique_locations} different locations in short time',
                'severity': 'high',
                'risk_contribution': 25
            })
            risk_score += 25
        
        return flags, risk_score
    
    def check_impossible_travel(self, transaction):
        """Check for impossible travel (transaction from distant location too quickly)"""
        user_id = transaction['user_id']
        current_time = transaction.get('timestamp', datetime.now())
        current_lat = transaction.get('latitude')
        current_lon = transaction.get('longitude')
        
        if current_lat is None or current_lon is None:
            return [], 0
        
        flags = []
        risk_score = 0
        
        # Get last known location
        user_locs = list(self.user_locations[user_id])
        if len(user_locs) > 0:
            last_loc = user_locs[-1]
            
            # Calculate distance
            distance = self._haversine_distance(
                last_loc['lat'], last_loc['lon'],
                current_lat, current_lon
            )
            
            # Calculate time difference
            time_diff_hours = (current_time - last_loc['timestamp']).total_seconds() / 3600
            
            # Check if travel is possible (assuming 500 km/h max speed)
            max_possible_distance = time_diff_hours * 500
            
            if distance > max_possible_distance and distance > 100:  # More than 100 km
                flags.append({
                    'type': 'impossible_travel',
                    'description': f'{distance:.0f} km in {time_diff_hours:.1f} hours',
                    'severity': 'critical',
                    'risk_contribution': 40
                })
                risk_score += 40
        
        return flags, risk_score
    
    def check_amount_anomaly(self, transaction):
        """Check for unusual transaction amounts"""
        user_id = transaction['user_id']
        amount = transaction['amount']
        
        flags = []
        risk_score = 0
        
        # Get user's transaction history
        user_txns = list(self.user_transactions[user_id])
        
        if len(user_txns) < 3:
            # New user - be more cautious
            if amount > 500:
                flags.append({
                    'type': 'large_amount_new_user',
                    'description': f'Large transaction (${amount:.2f}) for new user',
                    'severity': 'medium',
                    'risk_contribution': 20
                })
                risk_score += 20
        else:
            # Calculate user's average and std
            amounts = [t['amount'] for t in user_txns]
            avg_amount = np.mean(amounts)
            std_amount = np.std(amounts)
            
            # Check if amount is anomalous (> 3 standard deviations)
            if std_amount > 0:
                z_score = (amount - avg_amount) / std_amount
                
                if z_score > 3:
                    flags.append({
                        'type': 'unusual_amount',
                        'description': f'Amount ${amount:.2f} is {z_score:.1f}σ above user average',
                        'severity': 'high',
                        'risk_contribution': 25
                    })
                    risk_score += 25
                elif z_score > 2:
                    flags.append({
                        'type': 'elevated_amount',
                        'description': f'Amount ${amount:.2f} is {z_score:.1f}σ above user average',
                        'severity': 'medium',
                        'risk_contribution': 15
                    })
                    risk_score += 15
        
        return flags, risk_score
    
    def check_merchant_risk(self, transaction):
        """Check merchant category risk"""
        merchant = transaction.get('merchant_category', 'unknown')
        
        flags = []
        risk_score = 0
        
        # High-risk merchant categories
        high_risk_merchants = {
            'gambling': 25,
            'crypto': 30,
            'money_transfer': 20,
            'gift_cards': 15
        }
        
        if merchant in high_risk_merchants:
            risk_value = high_risk_merchants[merchant]
            flags.append({
                'type': 'high_risk_merchant',
                'description': f'Transaction at high-risk merchant: {merchant}',
                'severity': 'medium',
                'risk_contribution': risk_value
            })
            risk_score += risk_value
        
        # Check merchant fraud history
        merchant_data = self.merchant_stats[merchant]
        if merchant_data['count'] > 10:
            fraud_rate = merchant_data['fraud_count'] / merchant_data['count']
            if fraud_rate > 0.1:  # > 10% fraud rate
                flags.append({
                    'type': 'merchant_fraud_history',
                    'description': f'Merchant has {fraud_rate:.1%} fraud rate',
                    'severity': 'high',
                    'risk_contribution': 20
                })
                risk_score += 20
        
        return flags, risk_score
    
    def check_device_anomaly(self, transaction):
        """Check for device-based anomalies"""
        user_id = transaction['user_id']
        device_id = transaction.get('device_id')
        
        flags = []
        risk_score = 0
        
        if device_id:
            # Check if it's a new device
            if device_id not in self.user_devices[user_id]:
                flags.append({
                    'type': 'new_device',
                    'description': 'Transaction from previously unseen device',
                    'severity': 'medium',
                    'risk_contribution': 15
                })
                risk_score += 15
        
        return flags, risk_score
    
    def check_time_anomaly(self, transaction):
        """Check for unusual transaction times"""
        timestamp = transaction.get('timestamp', datetime.now())
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        flags = []
        risk_score = 0
        
        # Check for late night transactions (10 PM - 6 AM)
        if hour >= 22 or hour <= 6:
            flags.append({
                'type': 'unusual_time',
                'description': f'Transaction at {hour}:00 (late night/early morning)',
                'severity': 'low',
                'risk_contribution': 10
            })
            risk_score += 10
        
        return flags, risk_score
    
    def monitor_transaction(self, transaction):
        """Comprehensive real-time monitoring of a transaction"""
        logger.info(f"Monitoring transaction: {transaction.get('transaction_id', 'N/A')}")
        
        start_time = time.time()
        
        # Run all checks
        all_flags = []
        total_risk = 0
        
        checks = [
            self.check_velocity_attack,
            self.check_impossible_travel,
            self.check_amount_anomaly,
            self.check_merchant_risk,
            self.check_device_anomaly,
            self.check_time_anomaly
        ]
        
        for check in checks:
            flags, risk = check(transaction)
            all_flags.extend(flags)
            total_risk += risk
        
        # Add transaction to history
        self.add_transaction(transaction)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # in ms
        
        # Cap risk score at 100
        total_risk = min(total_risk, 100)
        
        result = {
            'transaction_id': transaction.get('transaction_id', 'N/A'),
            'risk_score': total_risk,
            'flags': all_flags,
            'processing_time_ms': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Determine decision
        if total_risk >= 70:
            result['decision'] = 'block'
            result['action'] = 'Transaction blocked - High fraud risk'
        elif total_risk >= 40:
            result['decision'] = 'challenge'
            result['action'] = 'Additional verification required (MFA/3DS)'
        else:
            result['decision'] = 'approve'
            result['action'] = 'Transaction approved'
        
        logger.info(f"Risk Score: {total_risk} | Decision: {result['decision']} | Time: {processing_time:.2f}ms")
        
        return result
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def get_user_summary(self, user_id):
        """Get summary statistics for a user"""
        txns = list(self.user_transactions[user_id])
        
        if not txns:
            return None
        
        amounts = [t['amount'] for t in txns]
        
        summary = {
            'user_id': user_id,
            'total_transactions': len(txns),
            'total_amount': sum(amounts),
            'avg_amount': np.mean(amounts),
            'median_amount': np.median(amounts),
            'max_amount': max(amounts),
            'unique_devices': len(self.user_devices[user_id]),
            'unique_merchants': len(set(t.get('merchant') for t in txns if t.get('merchant'))),
        }
        
        return summary


def simulate_real_time_monitoring():
    """Simulate real-time transaction monitoring"""
    monitor = TransactionMonitor()
    
    # Simulate some normal transactions
    logger.info("\n" + "="*70)
    logger.info("SIMULATING NORMAL TRANSACTIONS")
    logger.info("="*70)
    
    normal_txns = [
        {
            'transaction_id': 'TXN001',
            'user_id': 'USER123',
            'amount': 45.99,
            'merchant_category': 'grocery',
            'location': 'US',
            'device_id': 'DEVICE_A',
            'latitude': 40.7128,
            'longitude': -74.0060,
            'timestamp': datetime.now()
        },
        {
            'transaction_id': 'TXN002',
            'user_id': 'USER123',
            'amount': 89.50,
            'merchant_category': 'restaurant',
            'location': 'US',
            'device_id': 'DEVICE_A',
            'latitude': 40.7580,
            'longitude': -73.9855,
            'timestamp': datetime.now() + timedelta(hours=2)
        }
    ]
    
    for txn in normal_txns:
        result = monitor.monitor_transaction(txn)
        print(f"\n{result}")
    
    # Simulate fraudulent transactions
    logger.info("\n" + "="*70)
    logger.info("SIMULATING FRAUDULENT TRANSACTIONS")
    logger.info("="*70)
    
    fraud_txns = [
        {
            'transaction_id': 'TXN003',
            'user_id': 'USER123',
            'amount': 1500.00,  # Large amount
            'merchant_category': 'crypto',  # High-risk
            'location': 'RU',  # Foreign location
            'device_id': 'DEVICE_B',  # New device
            'latitude': 55.7558,
            'longitude': 37.6173,
            'timestamp': datetime.now() + timedelta(hours=2, minutes=30)  # Impossible travel
        },
        {
            'transaction_id': 'TXN004',
            'user_id': 'USER123',
            'amount': 2000.00,
            'merchant_category': 'gambling',
            'location': 'CN',
            'device_id': 'DEVICE_B',
            'latitude': 39.9042,
            'longitude': 116.4074,
            'timestamp': datetime.now() + timedelta(hours=2, minutes=35)  # Velocity attack
        }
    ]
    
    for txn in fraud_txns:
        result = monitor.monitor_transaction(txn)
        print(f"\n{result}")
    
    # Print user summary
    logger.info("\n" + "="*70)
    logger.info("USER SUMMARY")
    logger.info("="*70)
    summary = monitor.get_user_summary('USER123')
    print(summary)


if __name__ == "__main__":
    simulate_real_time_monitoring()
