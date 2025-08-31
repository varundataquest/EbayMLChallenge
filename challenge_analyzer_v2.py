#!/usr/bin/env python3
"""
Enhanced EvalAI Challenge Analyzer
This script provides multiple methods to understand the challenge requirements.
"""

import requests
import json
import logging
from bs4 import BeautifulSoup
from pathlib import Path
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedChallengeAnalyzer:
    """Enhanced analyzer for EvalAI challenges."""
    
    def __init__(self, challenge_id=2508):
        self.challenge_id = challenge_id
        self.base_url = "https://eval.ai"
        self.challenge_url = f"{self.base_url}/web/challenges/challenge-page/{challenge_id}/overview"
        self.api_url = f"{self.base_url}/api/challenges/{challenge_id}/"
        
    def fetch_with_session(self):
        """Fetch challenge page with session handling."""
        try:
            session = requests.Session()
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # First, try to access the main page
            logger.info(f"Fetching challenge page: {self.challenge_url}")
            response = session.get(self.challenge_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                logger.info("Successfully fetched challenge page")
                return response.text
            else:
                logger.warning(f"HTTP {response.status_code}: {response.reason}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Error fetching challenge page: {e}")
            return None
    
    def try_api_access(self):
        """Try to access the challenge via API."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'application/json',
            }
            
            logger.info(f"Trying API access: {self.api_url}")
            response = requests.get(self.api_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                logger.info("Successfully accessed challenge API")
                return response.json()
            else:
                logger.warning(f"API access failed: HTTP {response.status_code}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Error accessing API: {e}")
            return None
    
    def extract_from_html(self, html_content):
        """Extract information from HTML content using multiple strategies."""
        if not html_content:
            return {}
            
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Strategy 1: Look for common patterns
        info = {}
        
        # Try to find title
        title_selectors = [
            'h1', 'title', '.challenge-title', '.title', 
            '[class*="title"]', '[id*="title"]'
        ]
        
        for selector in title_selectors:
            elem = soup.select_one(selector)
            if elem and elem.get_text().strip():
                info['title'] = elem.get_text().strip()
                break
        
        # Try to find description
        desc_selectors = [
            '.description', '.challenge-description', '.desc',
            '[class*="description"]', '[id*="description"]',
            'p', '.content'
        ]
        
        for selector in desc_selectors:
            elem = soup.select_one(selector)
            if elem and elem.get_text().strip():
                text = elem.get_text().strip()
                if len(text) > 50:  # Likely a description
                    info['description'] = text
                    break
        
        # Strategy 2: Look for specific keywords
        text_content = soup.get_text()
        
        # Look for evaluation metrics
        metric_patterns = [
            r'evaluation metric[s]?[:\s]+([^.\n]+)',
            r'metric[s]?[:\s]+([^.\n]+)',
            r'score[:\s]+([^.\n]+)',
            r'accuracy[:\s]+([^.\n]+)',
            r'precision[:\s]+([^.\n]+)',
            r'recall[:\s]+([^.\n]+)',
            r'f1[:\s]+([^.\n]+)',
        ]
        
        for pattern in metric_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                info['evaluation_metric'] = match.group(1).strip()
                break
        
        # Look for data format information
        data_patterns = [
            r'data format[:\s]+([^.\n]+)',
            r'input format[:\s]+([^.\n]+)',
            r'file format[:\s]+([^.\n]+)',
            r'\.csv[:\s]+([^.\n]+)',
            r'\.json[:\s]+([^.\n]+)',
        ]
        
        for pattern in data_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                info['data_format'] = match.group(1).strip()
                break
        
        return info
    
    def create_manual_analysis_guide(self):
        """Create a guide for manual analysis."""
        guide_file = Path(f"manual_analysis_guide_{self.challenge_id}.md")
        
        with open(guide_file, 'w') as f:
            f.write(f"# Manual Analysis Guide for EvalAI Challenge #{self.challenge_id}\n\n")
            f.write("## Challenge URL\n")
            f.write(f"{self.challenge_url}\n\n")
            
            f.write("## Steps to Analyze the Challenge\n\n")
            f.write("1. **Visit the Challenge Page**\n")
            f.write(f"   - Go to: {self.challenge_url}\n")
            f.write("   - You may need to log in to EvalAI\n\n")
            
            f.write("2. **Key Information to Look For**\n")
            f.write("   - Challenge title and description\n")
            f.write("   - Dataset information and download links\n")
            f.write("   - Evaluation metrics\n")
            f.write("   - Submission format requirements\n")
            f.write("   - Timeline and deadlines\n")
            f.write("   - Prize information\n\n")
            
            f.write("3. **Technical Requirements**\n")
            f.write("   - Programming languages allowed\n")
            f.write("   - Libraries and frameworks\n")
            f.write("   - Hardware requirements\n")
            f.write("   - Submission size limits\n\n")
            
            f.write("4. **Data Understanding**\n")
            f.write("   - Data format (CSV, JSON, images, etc.)\n")
            f.write("   - Feature descriptions\n")
            f.write("   - Target variable\n")
            f.write("   - Data size and structure\n\n")
            
            f.write("5. **Next Steps**\n")
            f.write("   - Download the dataset\n")
            f.write("   - Explore the data\n")
            f.write("   - Implement baseline solution\n")
            f.write("   - Iterate and improve\n\n")
            
            f.write("## Common Challenge Types\n")
            f.write("- **Classification**: Predict categories/classes\n")
            f.write("- **Regression**: Predict continuous values\n")
            f.write("- **Object Detection**: Locate objects in images\n")
            f.write("- **Natural Language Processing**: Text analysis\n")
            f.write("- **Recommendation Systems**: Suggest items\n")
            f.write("- **Time Series**: Predict future values\n\n")
            
            f.write("## Useful Tools and Libraries\n")
            f.write("- **Data Analysis**: pandas, numpy, matplotlib, seaborn\n")
            f.write("- **Machine Learning**: scikit-learn, xgboost, lightgbm\n")
            f.write("- **Deep Learning**: tensorflow, pytorch, keras\n")
            f.write("- **Computer Vision**: opencv, pillow\n")
            f.write("- **NLP**: nltk, spacy, transformers\n")
        
        logger.info(f"Manual analysis guide saved to {guide_file}")
    
    def analyze(self):
        """Main analysis method."""
        logger.info(f"Starting enhanced analysis of EvalAI Challenge #{self.challenge_id}")
        
        # Try API access first
        api_data = self.try_api_access()
        if api_data:
            logger.info("Found API data:")
            logger.info(json.dumps(api_data, indent=2))
            
            # Save API data
            with open(f"challenge_{self.challenge_id}_api.json", 'w') as f:
                json.dump(api_data, f, indent=2)
        
        # Try web page access
        html_content = self.fetch_with_session()
        if html_content:
            # Save raw HTML for inspection
            with open(f"challenge_{self.challenge_id}_page.html", 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Extract information
            extracted_info = self.extract_from_html(html_content)
            
            # Save extracted info
            with open(f"challenge_{self.challenge_id}_extracted.json", 'w') as f:
                json.dump(extracted_info, f, indent=2)
            
            logger.info("Extracted information:")
            logger.info(json.dumps(extracted_info, indent=2))
        
        # Create manual analysis guide
        self.create_manual_analysis_guide()
        
        logger.info("Analysis complete. Check the generated files for details.")
        logger.info("If automated extraction didn't work, use the manual analysis guide.")

def main():
    """Main entry point."""
    analyzer = EnhancedChallengeAnalyzer()
    analyzer.analyze()

if __name__ == "__main__":
    main() 