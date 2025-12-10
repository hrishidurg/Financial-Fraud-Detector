"""
Risk Scoring Engine for Financial Fraud Detection
Combines ML predictions with rule-based scoring for final risk assessment
"""

import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskScoringEngine:
    """
    Advanced risk scoring engine that combines:
    - ML model predictions
    - Rule-based risk factors
    - Behavioral analysis
    - Explainability features
    """
    
    def __init__(self, model=None):
        self.model = model
        self.risk_weights = {
            'ml_score': 0.40,
            'amount_risk': 0.15,
            'location_risk': 0.15,
            'merchant_risk': 0.10,
            'device_risk': 0.10,
            'behavioral_risk': 0.10
        }
        self.thresholds = {
            'low': 30,
            'medium': 70,
            'high': 100
        }
    
    def calculate_ml_score(self, features):
        """Get ML model prediction score (0-100)"""
        if self.model is None:
            logger.warning("No ML model loaded, returning default score")
            return 50, []
        
        try:
            # Get prediction probability
            if hasattr(self.model, 'predict_ensemble'):
                ml_prob = self.model.predict_ensemble(features, method='weighted')
            elif hasattr(self.model, 'predict_proba'):
                ml_prob = self.model.predict_proba(features)[:, 1]
            else:
                logger.warning("Model doesn't support probability predictions")
                return 50, []
            
            ml_score = float(ml_prob[0] * 100)
            
            factors = [{
                'factor': 'ml_model_prediction',
                'contribution': ml_score * self.risk_weights['ml_score'],
                'confidence': float(ml_prob[0])
            }]
            
            return ml_score, factors
            
        except Exception as e:
            logger.error(f"Error in ML scoring: {e}")
            return 50, []
    
    def calculate_amount_risk(self, transaction, user_history=None):
        """Calculate risk based on transaction amount"""
        amount = transaction['amount']
        risk_score = 0
        factors = []
        
        # Base risk for large amounts
        if amount > 5000:
            risk_score += 40
            factors.append({
                'factor': 'very_large_amount',
                'value': f'${amount:.2f}',
                'contribution': 40 * self.risk_weights['amount_risk']
            })
        elif amount > 1000:
            risk_score += 25
            factors.append({
                'factor': 'large_amount',
                'value': f'${amount:.2f}',
                'contribution': 25 * self.risk_weights['amount_risk']
            })
        
        # Compare to user history if available
        if user_history and len(user_history) > 0:
            avg_amount = np.mean([t['amount'] for t in user_history])
            std_amount = np.std([t['amount'] for t in user_history])
            
            if std_amount > 0:
                z_score = (amount - avg_amount) / std_amount
                
                if z_score > 3:
                    risk_score += 35
                    factors.append({
                        'factor': 'amount_deviation',
                        'value': f'{z_score:.1f}σ above average',
                        'contribution': 35 * self.risk_weights['amount_risk']
                    })
                elif z_score > 2:
                    risk_score += 20
                    factors.append({
                        'factor': 'elevated_amount',
                        'value': f'{z_score:.1f}σ above average',
                        'contribution': 20 * self.risk_weights['amount_risk']
                    })
        
        return min(risk_score, 100), factors
    
    def calculate_location_risk(self, transaction, user_history=None):
        """Calculate risk based on location"""
        location = transaction.get('location', 'unknown')
        risk_score = 0
        factors = []
        
        # High-risk countries
        high_risk_countries = ['RU', 'CN', 'NG', 'PK', 'VN']
        if location in high_risk_countries:
            risk_score += 30
            factors.append({
                'factor': 'high_risk_location',
                'value': location,
                'contribution': 30 * self.risk_weights['location_risk']
            })
        
        # Check for foreign transaction if user history available
        if user_history and len(user_history) > 0:
            common_locations = [t.get('location') for t in user_history]
            if location not in common_locations:
                risk_score += 20
                factors.append({
                    'factor': 'unfamiliar_location',
                    'value': location,
                    'contribution': 20 * self.risk_weights['location_risk']
                })
        
        # Check for impossible travel (if lat/lon available)
        if 'latitude' in transaction and 'longitude' in transaction and user_history:
            for prev_txn in reversed(user_history[-5:]):  # Check last 5 transactions
                if 'latitude' in prev_txn and 'longitude' in prev_txn:
                    distance = self._haversine_distance(
                        prev_txn['latitude'], prev_txn['longitude'],
                        transaction['latitude'], transaction['longitude']
                    )
                    
                    time_diff = (transaction.get('timestamp', datetime.now()) - 
                               prev_txn.get('timestamp', datetime.now())).total_seconds() / 3600
                    
                    if time_diff > 0:
                        speed = distance / time_diff
                        if speed > 500:  # Faster than airplane
                            risk_score += 40
                            factors.append({
                                'factor': 'impossible_travel',
                                'value': f'{distance:.0f}km in {time_diff:.1f}h',
                                'contribution': 40 * self.risk_weights['location_risk']
                            })
                            break
        
        return min(risk_score, 100), factors
    
    def calculate_merchant_risk(self, transaction):
        """Calculate risk based on merchant"""
        merchant = transaction.get('merchant_category', 'unknown')
        risk_score = 0
        factors = []
        
        # High-risk merchant categories
        high_risk_merchants = {
            'crypto': 40,
            'gambling': 35,
            'money_transfer': 30,
            'gift_cards': 25,
            'wire_transfer': 30
        }
        
        if merchant in high_risk_merchants:
            risk_value = high_risk_merchants[merchant]
            risk_score += risk_value
            factors.append({
                'factor': 'high_risk_merchant',
                'value': merchant,
                'contribution': risk_value * self.risk_weights['merchant_risk']
            })
        
        return min(risk_score, 100), factors
    
    def calculate_device_risk(self, transaction, user_devices=None):
        """Calculate risk based on device"""
        device_id = transaction.get('device_id')
        risk_score = 0
        factors = []
        
        if device_id is None:
            risk_score += 15
            factors.append({
                'factor': 'no_device_fingerprint',
                'value': 'unknown',
                'contribution': 15 * self.risk_weights['device_risk']
            })
        elif user_devices and device_id not in user_devices:
            risk_score += 25
            factors.append({
                'factor': 'new_device',
                'value': device_id,
                'contribution': 25 * self.risk_weights['device_risk']
            })
        
        return min(risk_score, 100), factors
    
    def calculate_behavioral_risk(self, transaction, user_history=None):
        """Calculate risk based on behavioral patterns"""
        risk_score = 0
        factors = []
        
        # Time-based risk
        timestamp = transaction.get('timestamp', datetime.now())
        hour = timestamp.hour
        
        if hour >= 22 or hour <= 6:
            risk_score += 15
            factors.append({
                'factor': 'unusual_time',
                'value': f'{hour}:00',
                'contribution': 15 * self.risk_weights['behavioral_risk']
            })
        
        # Velocity check (if user history available)
        if user_history:
            recent_txns = [
                t for t in user_history
                if (timestamp - t.get('timestamp', datetime.now())).total_seconds() < 600
            ]
            
            if len(recent_txns) > 5:
                risk_score += 30
                factors.append({
                    'factor': 'high_velocity',
                    'value': f'{len(recent_txns)} txns in 10min',
                    'contribution': 30 * self.risk_weights['behavioral_risk']
                })
        
        return min(risk_score, 100), factors
    
    def calculate_risk_score(self, transaction, ml_features=None, user_history=None, user_devices=None):
        """
        Calculate comprehensive risk score
        
        Args:
            transaction: Transaction dictionary
            ml_features: Processed features for ML model
            user_history: User's transaction history
            user_devices: Set of user's known devices
            
        Returns:
            Dictionary with risk score, factors, and decision
        """
        logger.info(f"Calculating risk for transaction: {transaction.get('transaction_id', 'N/A')}")
        
        all_factors = []
        component_scores = {}
        
        # ML model score
        if ml_features is not None:
            ml_score, ml_factors = self.calculate_ml_score(ml_features)
            component_scores['ml_score'] = ml_score
            all_factors.extend(ml_factors)
        else:
            component_scores['ml_score'] = 50  # Default
        
        # Rule-based scores
        amount_score, amount_factors = self.calculate_amount_risk(transaction, user_history)
        component_scores['amount_risk'] = amount_score
        all_factors.extend(amount_factors)
        
        location_score, location_factors = self.calculate_location_risk(transaction, user_history)
        component_scores['location_risk'] = location_score
        all_factors.extend(location_factors)
        
        merchant_score, merchant_factors = self.calculate_merchant_risk(transaction)
        component_scores['merchant_risk'] = merchant_score
        all_factors.extend(merchant_factors)
        
        device_score, device_factors = self.calculate_device_risk(transaction, user_devices)
        component_scores['device_risk'] = device_score
        all_factors.extend(device_factors)
        
        behavioral_score, behavioral_factors = self.calculate_behavioral_risk(transaction, user_history)
        component_scores['behavioral_risk'] = behavioral_score
        all_factors.extend(behavioral_factors)
        
        # Calculate weighted final score
        final_score = sum(
            component_scores[component] * weight
            for component, weight in self.risk_weights.items()
        )
        
        final_score = min(final_score, 100)
        
        # Determine decision
        if final_score >= self.thresholds['medium']:
            decision = 'block'
            action = 'Transaction blocked - High fraud risk'
            confidence = 'high'
        elif final_score >= self.thresholds['low']:
            decision = 'challenge'
            action = 'Additional verification required (MFA/3DS)'
            confidence = 'medium'
        else:
            decision = 'approve'
            action = 'Transaction approved'
            confidence = 'high'
        
        # Sort factors by contribution
        all_factors.sort(key=lambda x: x.get('contribution', 0), reverse=True)
        
        result = {
            'transaction_id': transaction.get('transaction_id', 'N/A'),
            'risk_score': round(final_score, 2),
            'decision': decision,
            'action': action,
            'confidence': confidence,
            'component_scores': component_scores,
            'top_risk_factors': all_factors[:5],
            'all_factors': all_factors,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Risk Score: {final_score:.2f} | Decision: {decision}")
        
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
    
    def explain_score(self, result):
        """Generate human-readable explanation of risk score"""
        explanation = f"""
Risk Assessment Report
{'='*60}
Transaction ID: {result['transaction_id']}
Risk Score: {result['risk_score']}/100
Decision: {result['decision'].upper()}
Action: {result['action']}
Confidence: {result['confidence'].upper()}

Component Scores:
{'-'*60}
"""
        for component, score in result['component_scores'].items():
            weight = self.risk_weights[component]
            weighted_score = score * weight
            explanation += f"{component.replace('_', ' ').title()}: {score:.1f}/100 (weighted: {weighted_score:.1f})\n"
        
        explanation += f"\nTop Risk Factors:\n{'-'*60}\n"
        for i, factor in enumerate(result['top_risk_factors'], 1):
            factor_name = factor['factor'].replace('_', ' ').title()
            contribution = factor.get('contribution', 0)
            value = factor.get('value', 'N/A')
            explanation += f"{i}. {factor_name}: +{contribution:.1f} points (Value: {value})\n"
        
        return explanation


def demo_risk_scoring():
    """Demonstrate risk scoring engine"""
    engine = RiskScoringEngine()
    
    # Test case 1: Normal transaction
    logger.info("\n" + "="*70)
    logger.info("TEST CASE 1: NORMAL TRANSACTION")
    logger.info("="*70)
    
    normal_txn = {
        'transaction_id': 'TXN001',
        'amount': 49.99,
        'merchant_category': 'grocery',
        'location': 'US',
        'device_id': 'DEVICE_A',
        'latitude': 40.7128,
        'longitude': -74.0060,
        'timestamp': datetime.now()
    }
    
    result1 = engine.calculate_risk_score(normal_txn)
    print(engine.explain_score(result1))
    
    # Test case 2: Suspicious transaction
    logger.info("\n" + "="*70)
    logger.info("TEST CASE 2: SUSPICIOUS TRANSACTION")
    logger.info("="*70)
    
    suspicious_txn = {
        'transaction_id': 'TXN002',
        'amount': 2500.00,
        'merchant_category': 'crypto',
        'location': 'CN',
        'device_id': 'DEVICE_NEW',
        'latitude': 39.9042,
        'longitude': 116.4074,
        'timestamp': datetime.now()
    }
    
    result2 = engine.calculate_risk_score(suspicious_txn)
    print(engine.explain_score(result2))
    
    # Test case 3: With user history
    logger.info("\n" + "="*70)
    logger.info("TEST CASE 3: TRANSACTION WITH USER HISTORY")
    logger.info("="*70)
    
    user_history = [
        {'amount': 50, 'location': 'US', 'timestamp': datetime.now() - pd.Timedelta(days=1)},
        {'amount': 75, 'location': 'US', 'timestamp': datetime.now() - pd.Timedelta(days=2)},
        {'amount': 30, 'location': 'US', 'timestamp': datetime.now() - pd.Timedelta(days=3)}
    ]
    
    anomaly_txn = {
        'transaction_id': 'TXN003',
        'amount': 5000.00,
        'merchant_category': 'gambling',
        'location': 'RU',
        'device_id': None,
        'timestamp': datetime.now()
    }
    
    result3 = engine.calculate_risk_score(anomaly_txn, user_history=user_history)
    print(engine.explain_score(result3))


if __name__ == "__main__":
    demo_risk_scoring()
