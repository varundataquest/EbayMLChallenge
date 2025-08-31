"""
Configuration settings for EvalAI Challenge #2508
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, SUBMISSIONS_DIR]:
    directory.mkdir(exist_ok=True)

# Challenge configuration
CHALLENGE_ID = 2508
CHALLENGE_URL = "https://eval.ai/web/challenges/challenge-page/2508/overview"

# Model configuration
MODEL_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "validation_size": 0.1,
}

# Data configuration
DATA_CONFIG = {
    "train_file": "train.csv",
    "test_file": "test.csv",
    "sample_submission": "sample_submission.csv",
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001,
    "early_stopping_patience": 10,
}

# Environment variables
EVALAI_API_TOKEN = os.getenv("EVALAI_API_TOKEN")
EVALAI_PARTICIPATION_TOKEN = os.getenv("EVALAI_PARTICIPATION_TOKEN")

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "challenge.log",
} 