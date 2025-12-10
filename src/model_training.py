"""
ML Model Training Module for Financial Fraud Detection
Trains ensemble models using Watson Machine Learning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """Ensemble model for fraud detection"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.threshold = 0.5
        
    def train_random_forest(self, X_train, y_train, **kwargs):
        """Train Random Forest classifier"""
        logger.info("Training Random Forest model...")
        
        model = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', 20),
            min_samples_split=kwargs.get('min_samples_split', 10),
            min_samples_leaf=kwargs.get('min_samples_leaf', 4),
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        
        logger.info("Random Forest model trained successfully")
        return model
    
    def train_xgboost(self, X_train, y_train, **kwargs):
        """Train XGBoost classifier"""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, skipping...")
            return None
            
        logger.info("Training XGBoost model...")
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        
        model = xgb.XGBClassifier(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', 10),
            learning_rate=kwargs.get('learning_rate', 0.1),
            subsample=kwargs.get('subsample', 0.8),
            colsample_bytree=kwargs.get('colsample_bytree', 0.8),
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        
        logger.info("XGBoost model trained successfully")
        return model
    
    def train_gradient_boosting(self, X_train, y_train, **kwargs):
        """Train Gradient Boosting classifier"""
        logger.info("Training Gradient Boosting model...")
        
        model = GradientBoostingClassifier(
            n_estimators=kwargs.get('n_estimators', 150),
            max_depth=kwargs.get('max_depth', 8),
            learning_rate=kwargs.get('learning_rate', 0.1),
            subsample=kwargs.get('subsample', 0.8),
            random_state=42
        )
        
        model.fit(X_train, y_train)
        self.models['gradient_boosting'] = model
        
        logger.info("Gradient Boosting model trained successfully")
        return model
    
    def train_isolation_forest(self, X_train, contamination=0.02):
        """Train Isolation Forest for anomaly detection"""
        logger.info("Training Isolation Forest model...")
        
        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train)
        self.models['isolation_forest'] = model
        
        logger.info("Isolation Forest model trained successfully")
        return model
    
    def train_one_class_svm(self, X_train, nu=0.02):
        """Train One-Class SVM for anomaly detection"""
        logger.info("Training One-Class SVM model...")
        
        model = OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=nu
        )
        
        model.fit(X_train)
        self.models['one_class_svm'] = model
        
        logger.info("One-Class SVM model trained successfully")
        return model
    
    def train_ensemble(self, X_train, y_train, X_val=None, y_val=None):
        """Train ensemble of multiple models"""
        logger.info("Training ensemble of models...")
        
        # Train supervised models
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        
        # Train unsupervised models
        contamination = len(y_train[y_train == 1]) / len(y_train)
        self.train_isolation_forest(X_train, contamination=contamination)
        self.train_one_class_svm(X_train, nu=contamination)
        
        # Evaluate and select best model
        if X_val is not None and y_val is not None:
            self.evaluate_models(X_val, y_val)
        
        logger.info("Ensemble training completed")
        
    def predict_ensemble(self, X, method='voting'):
        """Predict using ensemble of models"""
        predictions = {}
        
        # Get predictions from supervised models
        for name in ['random_forest', 'xgboost', 'gradient_boosting']:
            if name in self.models:
                predictions[name] = self.models[name].predict_proba(X)[:, 1]
        
        # Get predictions from unsupervised models
        if 'isolation_forest' in self.models:
            # Convert -1/1 to 0/1 probabilities
            if_pred = self.models['isolation_forest'].predict(X)
            predictions['isolation_forest'] = (if_pred == -1).astype(float)
        
        if 'one_class_svm' in self.models:
            # Convert -1/1 to 0/1 probabilities
            svm_pred = self.models['one_class_svm'].predict(X)
            predictions['one_class_svm'] = (svm_pred == -1).astype(float)
        
        # Combine predictions
        if method == 'voting':
            # Average all predictions
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
        elif method == 'weighted':
            # Weighted average (higher weight for supervised models)
            weights = {
                'random_forest': 0.25,
                'xgboost': 0.25,
                'gradient_boosting': 0.25,
                'isolation_forest': 0.15,
                'one_class_svm': 0.10
            }
            ensemble_pred = sum(predictions[k] * weights.get(k, 0.2) 
                              for k in predictions.keys())
        else:
            # Use best model only
            if self.best_model and self.best_model in self.models:
                ensemble_pred = self.models[self.best_model].predict_proba(X)[:, 1]
            else:
                ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        return ensemble_pred
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        logger.info("Evaluating models...")
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Evaluating {name.upper()}")
            logger.info(f"{'='*50}")
            
            try:
                if name in ['random_forest', 'xgboost', 'gradient_boosting']:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    y_pred = (y_pred_proba > self.threshold).astype(int)
                    
                    # Calculate metrics
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                    
                elif name in ['isolation_forest', 'one_class_svm']:
                    y_pred = model.predict(X_test)
                    y_pred = (y_pred == -1).astype(int)  # Convert to 0/1
                    
                    # For unsupervised, calculate precision/recall
                    auc_score = None
                
                # Classification report
                report = classification_report(y_test, y_pred, output_dict=True)
                cm = confusion_matrix(y_test, y_pred)
                
                results[name] = {
                    'auc': auc_score,
                    'precision': report['1']['precision'],
                    'recall': report['1']['recall'],
                    'f1': report['1']['f1-score'],
                    'confusion_matrix': cm
                }
                
                logger.info(f"AUC Score: {auc_score:.4f}" if auc_score else "AUC: N/A")
                logger.info(f"Precision: {report['1']['precision']:.4f}")
                logger.info(f"Recall: {report['1']['recall']:.4f}")
                logger.info(f"F1 Score: {report['1']['f1-score']:.4f}")
                logger.info(f"\nConfusion Matrix:\n{cm}")
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
        
        # Select best model based on F1 score
        best_f1 = 0
        for name, metrics in results.items():
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                self.best_model = name
        
        logger.info(f"\nBest model: {self.best_model} (F1: {best_f1:.4f})")
        
        return results
    
    def get_feature_importance(self, feature_names, top_n=20):
        """Get feature importance from tree-based models"""
        if 'random_forest' in self.models:
            model = self.models['random_forest']
            importance = model.feature_importances_
            
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            logger.info(f"\nTop {top_n} Important Features:")
            print(feature_imp.head(top_n))
            
            return feature_imp
        
        return None
    
    def save_models(self, output_dir='models'):
        """Save all trained models"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, model in self.models.items():
            filepath = f"{output_dir}/{name}_{timestamp}.joblib"
            joblib.dump(model, filepath)
            logger.info(f"Saved {name} to {filepath}")
        
        # Save metadata
        metadata = {
            'best_model': self.best_model,
            'threshold': self.threshold,
            'timestamp': timestamp
        }
        joblib.dump(metadata, f"{output_dir}/metadata_{timestamp}.joblib")
        logger.info("Model metadata saved")
    
    def load_models(self, model_dir='models', timestamp=None):
        """Load trained models"""
        import os
        import glob
        
        if timestamp:
            pattern = f"{model_dir}/*_{timestamp}.joblib"
        else:
            pattern = f"{model_dir}/*.joblib"
        
        model_files = glob.glob(pattern)
        
        for filepath in model_files:
            filename = os.path.basename(filepath)
            if 'metadata' in filename:
                metadata = joblib.load(filepath)
                self.best_model = metadata['best_model']
                self.threshold = metadata['threshold']
            else:
                name = filename.split('_')[0]
                self.models[name] = joblib.load(filepath)
                logger.info(f"Loaded {name} from {filepath}")
        
        logger.info("Models loaded successfully")


def main():
    """Main training pipeline"""
    from data_processing import TransactionDataProcessor, generate_sample_data
    
    # Generate or load data
    logger.info("Preparing data...")
    data = generate_sample_data(num_transactions=10000, fraud_ratio=0.02)
    
    # Process data
    processor = TransactionDataProcessor()
    data = processor.clean_data(data)
    features = processor.extract_features(data)
    labels = data['is_fraud']
    
    # Prepare training data
    X, y = processor.prepare_training_data(features, labels)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Fraud ratio: {y_train.mean():.2%}")
    
    # Train models
    fraud_model = FraudDetectionModel()
    fraud_model.train_ensemble(X_train, y_train, X_val, y_val)
    
    # Final evaluation on test set
    logger.info("\n" + "="*50)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("="*50)
    fraud_model.evaluate_models(X_test, y_test)
    
    # Feature importance
    fraud_model.get_feature_importance(features.columns)
    
    # Save models
    fraud_model.save_models()
    
    logger.info("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()
