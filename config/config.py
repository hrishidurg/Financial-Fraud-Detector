# IBM Watson Financial Fraud Detection - Configuration

# Watson Machine Learning Configuration
WATSON_ML_API_KEY = "your_watson_api_key_here"
WATSON_ML_URL = "https://us-south.ml.cloud.ibm.com"
WATSON_ML_SPACE_ID = "your_space_id_here"

# Model Configuration
MODEL_VERSION = "1.0.0"
MODEL_TYPE = "ensemble"
FRAUD_THRESHOLD_LOW = 30
FRAUD_THRESHOLD_HIGH = 70

# Risk Scoring Weights
RISK_WEIGHTS = {
    "ml_score": 0.40,
    "amount_risk": 0.15,
    "location_risk": 0.15,
    "merchant_risk": 0.10,
    "device_risk": 0.10,
    "behavioral_risk": 0.10
}

# High-Risk Configurations
HIGH_RISK_COUNTRIES = ["RU", "CN", "NG", "PK", "VN", "KP"]
HIGH_RISK_MERCHANTS = {
    "crypto": 40,
    "gambling": 35,
    "money_transfer": 30,
    "gift_cards": 25,
    "wire_transfer": 30
}

# Transaction Limits
MAX_TRANSACTION_AMOUNT = 10000
LARGE_TRANSACTION_THRESHOLD = 1000
VERY_LARGE_TRANSACTION_THRESHOLD = 5000

# Velocity Checks
VELOCITY_WINDOW_MINUTES = 10
MAX_TRANSACTIONS_PER_WINDOW = 5
MAX_LOCATIONS_PER_WINDOW = 3

# Monitoring Configuration
MONITOR_LOOKBACK_HOURS = 24
IMPOSSIBLE_TRAVEL_SPEED_KMH = 500  # Max realistic speed

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = False
API_VERSION = "v1"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "logs/fraud_detection.log"

# Database Configuration (if needed)
DATABASE_URL = "postgresql://user:password@localhost:5432/fraud_detection"
REDIS_URL = "redis://localhost:6379/0"

# Feature Engineering
FEATURE_COLUMNS = [
    "amount",
    "amount_log",
    "amount_zscore",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_night",
    "transactions_last_hour",
    "transactions_last_day",
    "merchant_category_encoded",
    "is_high_risk_merchant",
    "location_encoded",
    "is_new_device",
    "distance_from_last"
]

# Performance Targets
TARGET_LATENCY_MS = 200
TARGET_FRAUD_DETECTION_RATE = 0.95
TARGET_FALSE_POSITIVE_RATE = 0.05
TARGET_THROUGHPUT_TPS = 10000
