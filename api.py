"""
AI Research Paper Intelligence System - REST API Production Server
Author: Kislay Anand
"""

import os
import faiss
import torch
import pandas as pd
import nest_asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from keybert import KeyBERT

# --- Global Initialization & System Config ---
device = 0 if torch.cuda.is_available() else -1
DATA_DIR = "/content/drive/MyDrive/ai_research_pipeline"
INDEX_PATH = os.path.join(DATA_DIR, "production_paper_faiss.index")

print("Bootstrapping dataset reference...")
raw_dataset = load_dataset('CShorten/ML-ArXiv-Papers', split='train')
df = pd.DataFrame(raw_dataset)[['title', 'abstract']].dropna()
df['paper_text'] = df['title'] + " " + df['abstract']
df['paper_text'] = df['paper_text'].str.replace('\n', ' ', regex=False).str.strip()

print("Loading saved FAISS vector database index...")
if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError("FAISS index missing. Run build_index.py offline first!")
index = faiss.read_index(INDEX_PATH)

print("Spawning core ML intelligence pipelines...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda' if device == 0 else 'cpu')
summarizer = pipeline(task="summarization", model="facebook/bart-large-cnn", device=device)
ner_model = pipeline(task="ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=device)
kw_model = KeyBERT()

# --- FastAPI Implementation Layers ---
app = FastAPI(
    title="AI Research Paper Intelligence System",
    description="Production-grade semantic search, abstract summarization, and NER parsing API."
)

class SearchRequest(BaseModel):
    query: str = Field(..., example="deep learning for medical image analysis")
    top_k: int = Field(default=3, ge=1, le=10)

class EntityModel(BaseModel):
    entity: str
    word: str

class ResearchPaperResponse(BaseModel):
    title: str
    similarity_score: float
    summary: str
    keywords: List[str]
    entities: List[EntityModel]

@app.post("/api/v1/search", response_model=List[ResearchPaperResponse])
def search_and_analyze_papers(payload: SearchRequest):
    try:
        query_vector = embedding_model.encode([payload.query])
        faiss.normalize_L2(query_vector)
        
        scores, indices = index.search(query_vector, payload.top_k)
        
        payload_responses = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
                
            metadata = df.iloc[int(idx)]
            raw_text = metadata['paper_text']
            
            # 1. Summarization Task
            summary_payload = summarizer(raw_text, max_length=120, min_length=40, do_sample=False)
            extracted_summary = summary_payload[0]['summary_text']
            
            # 2. Keyphrase Task
            keywords_payload = kw_model.extract_keywords(raw_text, keyphrase_ngram_range=(1, 2), top_n=5)
            extracted_keywords = [kw[0] for kw in keywords_payload]
            
            # 3. Entity Classification Task
            ner_payload = ner_model(raw_text[:1024])
            extracted_entities = [
                EntityModel(entity=ent['entity_group'], word=ent['word']) 
                for ent in ner_payload
            ]
            
            payload_responses.append(
                ResearchPaperResponse(
                    title=metadata['title'],
                    similarity_score=float(score),
                    summary=extracted_summary,
                    keywords=extracted_keywords,
                    entities=extracted_entities
                )
            )
            
        return payload_responses

    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Internal Server Pipeline Error: {str(error)}")

if __name__ == "__main__":
    nest_asyncio.apply()
    print("Launching backend server application loop...")
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    # Using python's event loop to safely run when executed directly
    import asyncio
    asyncio.run(server.serve())
