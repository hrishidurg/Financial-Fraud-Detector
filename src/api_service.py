"""
Flask API Service for Financial Fraud Detection
REST API for real-time fraud detection
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import joblib
import os

from data_processing import TransactionDataProcessor
from model_training import FraudDetectionModel
from risk_scoring import RiskScoringEngine
from real_time_monitor import TransactionMonitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global instances
processor = TransactionDataProcessor()
fraud_model = FraudDetectionModel()
risk_engine = RiskScoringEngine()
monitor = TransactionMonitor()

# Model loaded flag
model_loaded = False


def load_models():
    """Load trained models on startup"""
    global model_loaded, fraud_model, risk_engine
    
    try:
        model_dir = 'models'
        if os.path.exists(model_dir):
            fraud_model.load_models(model_dir)
            risk_engine.model = fraud_model
            model_loaded = True
            logger.info("Models loaded successfully")
        else:
            logger.warning("No models found. Running in rule-based mode only.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.warning("Running in rule-based mode only.")


@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'service': 'Financial Fraud Detection API',
        'version': '1.0.0',
        'status': 'running',
        'model_loaded': model_loaded,
        'endpoints': {
            'health': '/health',
            'predict': '/api/v1/predict',
            'batch_predict': '/api/v1/predict/batch',
            'monitor': '/api/v1/monitor',
            'stats': '/api/v1/stats/user/<user_id>'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_loaded
    })


@app.route('/api/v1/predict', methods=['POST'])
def predict():
    """
    Predict fraud risk for a single transaction
    
    Request body:
    {
        "transaction_id": "TXN123",
        "user_id": "USER456",
        "amount": 150.00,
        "merchant_category": "online",
        "location": "US",
        "device_id": "DEVICE_ABC",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "timestamp": "2025-12-10T14:30:00"
    }
    """
    try:
        # Get transaction data
        transaction = request.get_json()
        
        if not transaction:
            return jsonify({'error': 'No transaction data provided'}), 400
        
        # Validate required fields
        required_fields = ['transaction_id', 'user_id', 'amount']
        missing_fields = [field for field in required_fields if field not in transaction]
        
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields
            }), 400
        
        # Add timestamp if not provided
        if 'timestamp' not in transaction:
            transaction['timestamp'] = datetime.now()
        else:
            transaction['timestamp'] = pd.to_datetime(transaction['timestamp'])
        
        logger.info(f"Processing transaction: {transaction['transaction_id']}")
        
        # Prepare features for ML model
        ml_features = None
        if model_loaded:
            try:
                ml_features = processor.prepare_inference_data(transaction)
            except Exception as e:
                logger.warning(f"Error preparing ML features: {e}")
        
        # Get user history from monitor
        user_id = transaction['user_id']
        user_history = list(monitor.user_transactions[user_id])
        user_devices = monitor.user_devices[user_id]
        
        # Calculate risk score
        result = risk_engine.calculate_risk_score(
            transaction=transaction,
            ml_features=ml_features,
            user_history=user_history,
            user_devices=user_devices
        )
        
        # Add transaction to monitor
        monitor.add_transaction(transaction)
        
        # Return result
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/api/v1/predict/batch', methods=['POST'])
def batch_predict():
    """
    Predict fraud risk for multiple transactions
    
    Request body:
    {
        "transactions": [
            { transaction_1 },
            { transaction_2 },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'transactions' not in data:
            return jsonify({'error': 'No transactions provided'}), 400
        
        transactions = data['transactions']
        results = []
        
        for txn in transactions:
            # Add timestamp if not provided
            if 'timestamp' not in txn:
                txn['timestamp'] = datetime.now()
            else:
                txn['timestamp'] = pd.to_datetime(txn['timestamp'])
            
            # Prepare features
            ml_features = None
            if model_loaded:
                try:
                    ml_features = processor.prepare_inference_data(txn)
                except Exception as e:
                    logger.warning(f"Error preparing ML features: {e}")
            
            # Get user history
            user_id = txn['user_id']
            user_history = list(monitor.user_transactions[user_id])
            user_devices = monitor.user_devices[user_id]
            
            # Calculate risk
            result = risk_engine.calculate_risk_score(
                transaction=txn,
                ml_features=ml_features,
                user_history=user_history,
                user_devices=user_devices
            )
            
            results.append(result)
            
            # Add to monitor
            monitor.add_transaction(txn)
        
        return jsonify({
            'total_transactions': len(transactions),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/api/v1/monitor', methods=['POST'])
def monitor_transaction():
    """
    Real-time monitoring of a transaction
    Checks for velocity attacks, impossible travel, etc.
    """
    try:
        transaction = request.get_json()
        
        if not transaction:
            return jsonify({'error': 'No transaction data provided'}), 400
        
        # Add timestamp if not provided
        if 'timestamp' not in transaction:
            transaction['timestamp'] = datetime.now()
        else:
            transaction['timestamp'] = pd.to_datetime(transaction['timestamp'])
        
        # Monitor transaction
        result = monitor.monitor_transaction(transaction)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in monitoring: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/api/v1/stats/user/<user_id>', methods=['GET'])
def get_user_stats(user_id):
    """Get statistics for a specific user"""
    try:
        summary = monitor.get_user_summary(user_id)
        
        if summary is None:
            return jsonify({
                'error': 'User not found',
                'user_id': user_id
            }), 404
        
        return jsonify(summary), 200
        
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/api/v1/explain', methods=['POST'])
def explain_prediction():
    """
    Get detailed explanation for a fraud prediction
    """
    try:
        data = request.get_json()
        
        if 'transaction' not in data:
            return jsonify({'error': 'No transaction provided'}), 400
        
        transaction = data['transaction']
        
        # Add timestamp if not provided
        if 'timestamp' not in transaction:
            transaction['timestamp'] = datetime.now()
        else:
            transaction['timestamp'] = pd.to_datetime(transaction['timestamp'])
        
        # Prepare features
        ml_features = None
        if model_loaded:
            try:
                ml_features = processor.prepare_inference_data(transaction)
            except Exception as e:
                logger.warning(f"Error preparing ML features: {e}")
        
        # Get user history
        user_id = transaction['user_id']
        user_history = list(monitor.user_transactions[user_id])
        user_devices = monitor.user_devices[user_id]
        
        # Calculate risk
        result = risk_engine.calculate_risk_score(
            transaction=transaction,
            ml_features=ml_features,
            user_history=user_history,
            user_devices=user_devices
        )
        
        # Generate explanation
        explanation = risk_engine.explain_score(result)
        
        return jsonify({
            'result': result,
            'explanation': explanation
        }), 200
        
    except Exception as e:
        logger.error(f"Error in explanation: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/api/v1/feedback', methods=['POST'])
def fraud_feedback():
    """
    Submit fraud feedback for model improvement
    
    Request body:
    {
        "transaction_id": "TXN123",
        "is_fraud": true,
        "notes": "Confirmed fraudulent transaction"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'transaction_id' not in data or 'is_fraud' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        transaction_id = data['transaction_id']
        is_fraud = data['is_fraud']
        notes = data.get('notes', '')
        
        # Log feedback
        logger.info(f"Fraud feedback received: TXN={transaction_id}, Fraud={is_fraud}")
        
        # In production, this would update the training data
        # For now, just acknowledge
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback received',
            'transaction_id': transaction_id,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Run the API
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Fraud Detection API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
