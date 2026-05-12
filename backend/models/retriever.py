"""Semantic search and passage retrieval"""
from sentence_transformers import SentenceTransformer, util
import torch
import aiohttp
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

class EvidenceRetriever:
    def __init__(self):
        logger.info("Loading retriever...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"✓ Retriever ready on {self.device}")
    
    async def extract_evidence(self, claim, sources, top_k=5):
        all_passages = []
        for source in sources:
            try:
                content = await self._fetch_page(source.get('url', ''))
                if not content:
                    all_passages.append({
                        'source_url': source['url'],
                        'source_title': source.get('title', 'Unknown'),
                        'excerpt': source.get('snippet', ''),
                        'text': source.get('snippet', '')
                    })
                    continue
                
                passages = self._split_into_passages(content)
                ranked = self._rank_passages(claim, passages, top_k)
                
                for p in ranked:
                    p.update({
                        'source_url': source['url'],
                        'source_title': source.get('title', 'Unknown')
                    })
                all_passages.extend(ranked)
            except Exception as e:
                logger.error(f"Error extracting from {source.get('url')}: {e}")
        
        return all_passages
    
    async def _fetch_page(self, url):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200: return ""
                    html = await resp.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            for elem in soup(['script', 'style']): elem.decompose()
            text = soup.get_text(separator=' ', strip=True)
            return ' '.join(text.split())[:10000]
        except: return ""
    
    def _split_into_passages(self, text, max_length=300):
        sentences = text.split('. ')
        passages = []
        current = []
        current_len = 0
        
        for sent in sentences:
            sent_len = len(sent.split())
            if current_len + sent_len > max_length:
                if current: passages.append('. '.join(current) + '.')
                current = [sent]
                current_len = sent_len
            else:
                current.append(sent)
                current_len += sent_len
        
        if current: passages.append('. '.join(current) + '.')
        return passages
    
    def _rank_passages(self, claim, passages, top_k=5):
        if not passages: return []
        claim_emb = self.embedder.encode(claim, convert_to_tensor=True)
        passage_embs = self.embedder.encode(passages, convert_to_tensor=True, show_progress_bar=False)
        sims = util.cos_sim(claim_emb, passage_embs)[0]
        top_idx = torch.argsort(sims, descending=True)[:top_k]
        return [{"text": passages[i], "excerpt": passages[i][:200] + "..."} for i in top_idx]
