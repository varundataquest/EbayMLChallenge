#!/usr/bin/env python3
"""
EvalAI Challenge #2508 - Solution Template
This template provides a structured approach to solving the challenge.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChallengeSolution:
    """Main solution class for EvalAI Challenge #2508."""
    
    def __init__(self):
        self.challenge_id = 2508
        self.data_dir = Path("data")
        self.models_dir = Path("models")
        self.results_dir = Path("results")
        self.submissions_dir = Path("submissions")
        
        # Create directories
        for directory in [self.data_dir, self.models_dir, self.results_dir, self.submissions_dir]:
            directory.mkdir(exist_ok=True)
        
        # Initialize data containers
        self.train_data = None
        self.test_data = None
        self.sample_submission = None
        self.model = None
        self.predictions = None
        
    def load_data(self):
        """Load the challenge dataset."""
        logger.info("Loading challenge data...")
        
        # Expected file paths - adjust based on actual challenge data
        train_path = self.data_dir / "train.csv"
        test_path = self.data_dir / "test.csv"
        sample_path = self.data_dir / "sample_submission.csv"
        
        try:
            # Load training data
            if train_path.exists():
                self.train_data = pd.read_csv(train_path)
                logger.info(f"Loaded training data: {self.train_data.shape}")
                logger.info(f"Columns: {list(self.train_data.columns)}")
            else:
                logger.warning(f"Training data not found at {train_path}")
                logger.info("Please download the training data from the challenge page")
            
            # Load test data
            if test_path.exists():
                self.test_data = pd.read_csv(test_path)
                logger.info(f"Loaded test data: {self.test_data.shape}")
            else:
                logger.warning(f"Test data not found at {test_path}")
            
            # Load sample submission
            if sample_path.exists():
                self.sample_submission = pd.read_csv(sample_path)
                logger.info(f"Loaded sample submission: {self.sample_submission.shape}")
            else:
                logger.warning(f"Sample submission not found at {sample_path}")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def explore_data(self):
        """Explore and understand the dataset."""
        if self.train_data is None:
            logger.warning("No training data loaded. Skipping exploration.")
            return
            
        logger.info("Exploring dataset...")
        
        # Basic information
        logger.info(f"Dataset shape: {self.train_data.shape}")
        logger.info(f"Data types:\n{self.train_data.dtypes}")
        
        # Check for missing values
        missing_values = self.train_data.isnull().sum()
        if missing_values.sum() > 0:
            logger.info(f"Missing values:\n{missing_values[missing_values > 0]}")
        else:
            logger.info("No missing values found")
        
        # Statistical summary
        logger.info("Statistical summary:")
        logger.info(self.train_data.describe())
        
        # Save exploration results
        exploration_file = self.results_dir / "data_exploration.txt"
        with open(exploration_file, 'w') as f:
            f.write("Dataset Exploration Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Shape: {self.train_data.shape}\n\n")
            f.write("Columns:\n")
            for col in self.train_data.columns:
                f.write(f"  - {col}\n")
            f.write(f"\nData Types:\n{self.train_data.dtypes}\n\n")
            f.write(f"Missing Values:\n{missing_values}\n\n")
            f.write("Statistical Summary:\n")
            f.write(self.train_data.describe().to_string())
        
        logger.info(f"Exploration results saved to {exploration_file}")
    
    def preprocess_data(self):
        """Preprocess the data for modeling."""
        if self.train_data is None:
            logger.warning("No training data loaded. Skipping preprocessing.")
            return
            
        logger.info("Preprocessing data...")
        
        # Make a copy to avoid modifying original data
        train_processed = self.train_data.copy()
        
        # Handle missing values
        if train_processed.isnull().sum().sum() > 0:
            # For numerical columns, fill with median
            numerical_cols = train_processed.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if train_processed[col].isnull().sum() > 0:
                    train_processed[col].fillna(train_processed[col].median(), inplace=True)
            
            # For categorical columns, fill with mode
            categorical_cols = train_processed.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if train_processed[col].isnull().sum() > 0:
                    train_processed[col].fillna(train_processed[col].mode()[0], inplace=True)
        
        # Encode categorical variables
        categorical_cols = train_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            train_processed[col] = train_processed[col].astype('category').cat.codes
        
        self.train_processed = train_processed
        logger.info("Data preprocessing completed")
    
    def prepare_features(self):
        """Prepare features for modeling."""
        if not hasattr(self, 'train_processed'):
            logger.warning("No processed data available. Skipping feature preparation.")
            return
            
        logger.info("Preparing features...")
        
        # Identify target variable (usually the last column or specified column)
        # This will need to be adjusted based on the actual challenge
        target_col = None
        
        # Common target column names
        possible_targets = ['target', 'label', 'class', 'y', 'outcome', 'result']
        for col in possible_targets:
            if col in self.train_processed.columns:
                target_col = col
                break
        
        if target_col is None:
            # Assume last column is target
            target_col = self.train_processed.columns[-1]
            logger.info(f"Using last column as target: {target_col}")
        
        # Separate features and target
        self.feature_cols = [col for col in self.train_processed.columns if col != target_col]
        self.X = self.train_processed[self.feature_cols]
        self.y = self.train_processed[target_col]
        
        logger.info(f"Features shape: {self.X.shape}")
        logger.info(f"Target shape: {self.y.shape}")
        logger.info(f"Target distribution:\n{self.y.value_counts()}")
    
    def train_model(self):
        """Train the model."""
        if not hasattr(self, 'X') or not hasattr(self, 'y'):
            logger.warning("Features not prepared. Skipping model training.")
            return
            
        logger.info("Training model...")
        
        # Split data
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Initialize and train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions on validation set
        val_predictions = self.model.predict(self.X_val)
        val_accuracy = accuracy_score(self.y_val, val_predictions)
        
        logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Save model
        model_file = self.models_dir / "baseline_model.pkl"
        import pickle
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Model saved to {model_file}")
        
        # Save validation results
        results_file = self.results_dir / "validation_results.txt"
        with open(results_file, 'w') as f:
            f.write("Validation Results\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Accuracy: {val_accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(self.y_val, val_predictions))
        
        logger.info(f"Validation results saved to {results_file}")
    
    def generate_submission(self):
        """Generate submission file."""
        if self.model is None:
            logger.warning("No trained model available. Skipping submission generation.")
            return
            
        if self.test_data is None:
            logger.warning("No test data available. Skipping submission generation.")
            return
            
        logger.info("Generating submission...")
        
        # Preprocess test data (apply same preprocessing as training data)
        test_processed = self.test_data.copy()
        
        # Handle missing values
        if test_processed.isnull().sum().sum() > 0:
            numerical_cols = test_processed.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if test_processed[col].isnull().sum() > 0:
                    test_processed[col].fillna(test_processed[col].median(), inplace=True)
            
            categorical_cols = test_processed.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if test_processed[col].isnull().sum() > 0:
                    test_processed[col].fillna(test_processed[col].mode()[0], inplace=True)
        
        # Encode categorical variables
        categorical_cols = test_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            test_processed[col] = test_processed[col].astype('category').cat.codes
        
        # Make predictions
        test_features = test_processed[self.feature_cols]
        self.predictions = self.model.predict(test_features)
        
        # Create submission file
        submission = pd.DataFrame({
            'id': range(len(self.predictions)),
            'prediction': self.predictions
        })
        
        # If sample submission exists, use its format
        if self.sample_submission is not None:
            submission = self.sample_submission.copy()
            submission.iloc[:, -1] = self.predictions  # Replace predictions column
        
        # Save submission
        submission_file = self.submissions_dir / "submission.csv"
        submission.to_csv(submission_file, index=False)
        
        logger.info(f"Submission saved to {submission_file}")
        logger.info(f"Submission shape: {submission.shape}")
    
    def run_pipeline(self):
        """Run the complete solution pipeline."""
        logger.info("Starting EvalAI Challenge solution pipeline...")
        
        try:
            self.load_data()
            self.explore_data()
            self.preprocess_data()
            self.prepare_features()
            self.train_model()
            self.generate_submission()
            
            logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            raise

def main():
    """Main entry point."""
    solution = ChallengeSolution()
    solution.run_pipeline()

if __name__ == "__main__":
    main() 