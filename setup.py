#!/usr/bin/env python3
"""
Setup script for EvalAI Challenge #2508
This script helps set up the project environment and download data.
"""

import os
import sys
import logging
from pathlib import Path
import requests
import zipfile
import tarfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChallengeSetup:
    """Setup class for EvalAI Challenge #2508."""
    
    def __init__(self):
        self.challenge_id = 2508
        self.challenge_url = f"https://eval.ai/web/challenges/challenge-page/{self.challenge_id}/overview"
        self.project_root = Path(__file__).parent
        
    def create_project_structure(self):
        """Create the project directory structure."""
        logger.info("Creating project directory structure...")
        
        directories = [
            "data",
            "models", 
            "results",
            "submissions",
            "logs",
            "notebooks",
            "src",
            "tests"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        
        # Create __init__.py files for Python packages
        init_files = ["src/__init__.py", "tests/__init__.py"]
        for init_file in init_files:
            init_path = self.project_root / init_file
            if not init_path.exists():
                init_path.touch()
                logger.info(f"Created file: {init_path}")
    
    def create_sample_data(self):
        """Create sample data files for testing."""
        logger.info("Creating sample data files...")
        
        import pandas as pd
        import numpy as np
        
        # Create sample training data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        # Generate sample features
        X = np.random.randn(n_samples, n_features)
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Generate sample target (binary classification)
        y = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        
        # Create training dataframe
        train_data = pd.DataFrame(X, columns=feature_names)
        train_data['target'] = y
        
        # Save training data
        train_file = self.project_root / "data" / "train.csv"
        train_data.to_csv(train_file, index=False)
        logger.info(f"Created sample training data: {train_file}")
        
        # Create sample test data
        n_test_samples = 200
        X_test = np.random.randn(n_test_samples, n_features)
        test_data = pd.DataFrame(X_test, columns=feature_names)
        
        # Save test data
        test_file = self.project_root / "data" / "test.csv"
        test_data.to_csv(test_file, index=False)
        logger.info(f"Created sample test data: {test_file}")
        
        # Create sample submission file
        sample_submission = pd.DataFrame({
            'id': range(n_test_samples),
            'prediction': np.random.choice([0, 1], size=n_test_samples)
        })
        
        submission_file = self.project_root / "data" / "sample_submission.csv"
        sample_submission.to_csv(submission_file, index=False)
        logger.info(f"Created sample submission file: {submission_file}")
    
    def create_environment_file(self):
        """Create .env file from template."""
        logger.info("Creating environment file...")
        
        env_template = self.project_root / "env_example.txt"
        env_file = self.project_root / ".env"
        
        if env_template.exists() and not env_file.exists():
            with open(env_template, 'r') as f:
                template_content = f.read()
            
            with open(env_file, 'w') as f:
                f.write(template_content)
            
            logger.info(f"Created environment file: {env_file}")
            logger.info("Please update the .env file with your actual credentials")
        else:
            logger.info("Environment file already exists or template not found")
    
    def create_jupyter_notebook(self):
        """Create a Jupyter notebook for exploration."""
        logger.info("Creating Jupyter notebook...")
        
        notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# EvalAI Challenge #2508 - Data Exploration\\n",
    "\\n",
    "This notebook helps you explore the challenge dataset and understand the problem."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "from pathlib import Path\\n",
    "\\n",
    "# Set up plotting\\n",
    "plt.style.use('seaborn-v0_8')\\n",
    "sns.set_palette(\"husl\")\\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load data\\n",
    "data_dir = Path('data')\\n",
    "\\n",
    "# Load training data\\n",
    "train_data = pd.read_csv(data_dir / 'train.csv')\\n",
    "print(f\"Training data shape: {train_data.shape}\")\\n",
    "print(f\"Columns: {list(train_data.columns)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Basic exploration\\n",
    "print(\"Data types:\")\\n",
    "print(train_data.dtypes)\\n",
    "print(\"\\nMissing values:\")\\n",
    "print(train_data.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Statistical summary\\n",
    "train_data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Target distribution\\n",
    "if 'target' in train_data.columns:\\n",
    "    plt.figure(figsize=(8, 6))\\n",
    "    train_data['target'].value_counts().plot(kind='bar')\\n",
    "    plt.title('Target Distribution')\\n",
    "    plt.xlabel('Target Class')\\n",
    "    plt.ylabel('Count')\\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature correlations\\n",
    "numerical_cols = train_data.select_dtypes(include=[np.number]).columns\\n",
    "if len(numerical_cols) > 1:\\n",
    "    plt.figure(figsize=(12, 8))\\n",
    "    sns.heatmap(train_data[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)\\n",
    "    plt.title('Feature Correlations')\\n",
    "    plt.tight_layout()\\n",
    "    plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
        
        notebook_file = self.project_root / "notebooks" / "01_data_exploration.ipynb"
        with open(notebook_file, 'w') as f:
            f.write(notebook_content)
        
        logger.info(f"Created Jupyter notebook: {notebook_file}")
    
    def print_next_steps(self):
        """Print next steps for the user."""
        logger.info("\\n" + "="*60)
        logger.info("SETUP COMPLETE!")
        logger.info("="*60)
        logger.info("\\nNext steps:")
        logger.info("1. Visit the challenge page: " + self.challenge_url)
        logger.info("2. Download the actual dataset from the challenge page")
        logger.info("3. Replace the sample data files in the 'data' directory")
        logger.info("4. Update the .env file with your EvalAI credentials")
        logger.info("5. Run the solution template: python solution_template.py")
        logger.info("6. Explore the data using the Jupyter notebook")
        logger.info("\\nGood luck with the challenge!")
    
    def setup(self):
        """Run the complete setup process."""
        logger.info("Setting up EvalAI Challenge #2508 project...")
        
        try:
            self.create_project_structure()
            self.create_sample_data()
            self.create_environment_file()
            self.create_jupyter_notebook()
            self.print_next_steps()
            
        except Exception as e:
            logger.error(f"Error during setup: {e}")
            raise

def main():
    """Main entry point."""
    setup = ChallengeSetup()
    setup.setup()

if __name__ == "__main__":
    main() 