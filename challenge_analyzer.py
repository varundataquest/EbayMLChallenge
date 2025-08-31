#!/usr/bin/env python3
"""
EvalAI Challenge Analyzer
This script helps understand the challenge requirements and structure.
"""

import requests
import json
import logging
from bs4 import BeautifulSoup
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChallengeAnalyzer:
    """Analyzes EvalAI challenge details."""
    
    def __init__(self, challenge_id=2508):
        self.challenge_id = challenge_id
        self.base_url = "https://eval.ai"
        self.challenge_url = f"{self.base_url}/web/challenges/challenge-page/{challenge_id}/overview"
        
    def fetch_challenge_page(self):
        """Fetch the challenge page content."""
        try:
            logger.info(f"Fetching challenge page: {self.challenge_url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(self.challenge_url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Error fetching challenge page: {e}")
            return None
    
    def parse_challenge_info(self, html_content):
        """Parse challenge information from HTML content."""
        if not html_content:
            return None
            
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract basic information
        challenge_info = {
            'title': None,
            'description': None,
            'evaluation_metric': None,
            'data_format': None,
            'submission_format': None,
            'timeline': None,
            'prizes': None
        }
        
        # Try to extract title
        title_elem = soup.find('h1') or soup.find('title')
        if title_elem:
            challenge_info['title'] = title_elem.get_text().strip()
        
        # Try to extract description
        desc_elem = soup.find('div', class_='description') or soup.find('p')
        if desc_elem:
            challenge_info['description'] = desc_elem.get_text().strip()
        
        # Look for specific sections
        sections = soup.find_all(['h2', 'h3', 'h4'])
        for section in sections:
            section_text = section.get_text().strip().lower()
            if 'metric' in section_text or 'evaluation' in section_text:
                next_elem = section.find_next_sibling()
                if next_elem:
                    challenge_info['evaluation_metric'] = next_elem.get_text().strip()
            
            elif 'data' in section_text or 'format' in section_text:
                next_elem = section.find_next_sibling()
                if next_elem:
                    challenge_info['data_format'] = next_elem.get_text().strip()
        
        return challenge_info
    
    def save_challenge_info(self, challenge_info):
        """Save challenge information to a JSON file."""
        if not challenge_info:
            return
            
        output_file = Path(f"challenge_{self.challenge_id}_info.json")
        with open(output_file, 'w') as f:
            json.dump(challenge_info, f, indent=2)
        
        logger.info(f"Challenge information saved to {output_file}")
    
    def create_challenge_summary(self, challenge_info):
        """Create a summary markdown file."""
        if not challenge_info:
            return
            
        summary_file = Path(f"challenge_{self.challenge_id}_summary.md")
        
        with open(summary_file, 'w') as f:
            f.write(f"# EvalAI Challenge #{self.challenge_id} Summary\n\n")
            
            if challenge_info['title']:
                f.write(f"## Title\n{challenge_info['title']}\n\n")
            
            if challenge_info['description']:
                f.write(f"## Description\n{challenge_info['description']}\n\n")
            
            if challenge_info['evaluation_metric']:
                f.write(f"## Evaluation Metric\n{challenge_info['evaluation_metric']}\n\n")
            
            if challenge_info['data_format']:
                f.write(f"## Data Format\n{challenge_info['data_format']}\n\n")
            
            f.write("## Next Steps\n")
            f.write("1. Review the challenge requirements\n")
            f.write("2. Download the dataset\n")
            f.write("3. Implement the solution\n")
            f.write("4. Submit results\n\n")
            
            f.write(f"## Challenge URL\n{self.challenge_url}\n")
        
        logger.info(f"Challenge summary saved to {summary_file}")
    
    def analyze(self):
        """Main analysis method."""
        logger.info(f"Starting analysis of EvalAI Challenge #{self.challenge_id}")
        
        # Fetch challenge page
        html_content = self.fetch_challenge_page()
        if not html_content:
            logger.error("Failed to fetch challenge page")
            return
        
        # Parse challenge information
        challenge_info = self.parse_challenge_info(html_content)
        if not challenge_info:
            logger.error("Failed to parse challenge information")
            return
        
        # Save information
        self.save_challenge_info(challenge_info)
        self.create_challenge_summary(challenge_info)
        
        # Print summary
        logger.info("Challenge Analysis Summary:")
        logger.info(f"Title: {challenge_info.get('title', 'Not found')}")
        description = challenge_info.get('description', 'Not found')
        if description and description != 'Not found':
            logger.info(f"Description: {description[:100]}...")
        else:
            logger.info(f"Description: {description}")
        
        return challenge_info

def main():
    """Main entry point."""
    analyzer = ChallengeAnalyzer()
    analyzer.analyze()

if __name__ == "__main__":
    main() 