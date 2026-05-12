"""Source credibility assessment"""
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

class CredibilityScorer:
    def __init__(self):
        self.domain_scores = {
            '.gov': 0.95, '.edu': 0.90,
            'pubmed': 0.95, 'nature': 0.95, 'science': 0.95,
            'nytimes': 0.80, 'reuters': 0.85,
            'wikipedia': 0.65
        }
    
    async def score(self, passage):
        url = passage.get('source_url', '').lower()
        base = 0.40
        for domain, score in self.domain_scores.items():
            if domain in url: return score
        return base
