import os
import numpy as np
import pandas as pd
import faiss
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def run_indexing_pipeline():
    # 1. Setup paths and hardware
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using Device: {'GPU (cuda)' if device == 0 else 'CPU'}")
    
    # Path inside Google Drive to persist data
    DATA_DIR = "/content/drive/MyDrive/ai_research_pipeline"
    os.makedirs(DATA_DIR, exist_ok=True)
    INDEX_PATH = os.path.join(DATA_DIR, "production_paper_faiss.index")

    # 2. Ingest Data
    print("Loading ArXiv Dataset...")
    raw_dataset = load_dataset('CShorten/ML-ArXiv-Papers', split='train')
    df = pd.DataFrame(raw_dataset)

    # Clean, structure, and optimize text lengths
    df = df[['title', 'abstract']].dropna()
    df['paper_text'] = df['title'] + " " + df['abstract']
    df['paper_text'] = df['paper_text'].str.replace('\n', ' ', regex=False).str.strip()
    print(f"Successfully processed {len(df)} records.")

    # 3. Initialize Embedding Engine
    print("Initializing Embedding Engine...")
    embedding_model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2', 
        device='cuda' if device == 0 else 'cpu'
    )

    # 4. Chunked Inference
    texts = df["paper_text"].tolist()
    chunk_size = 20000
    all_embeddings = []

    print(f"Executing batch generation for {len(texts)} documents...")
    for i in range(0, len(texts), chunk_size):
        chunk_file = os.path.join(DATA_DIR, f"chunk_{i}.npy")
        
        if os.path.exists(chunk_file):
            print(f"Found cache: Loading chunk index {i} from storage...")
            chunk_emb = np.load(chunk_file)
        else:
            print(f"Cache miss: Computing batch inference for chunk index {i}...")
            chunk_texts = texts[i : i + chunk_size]
            chunk_emb = embedding_model.encode(
                chunk_texts,
                batch_size=256,
                show_progress_bar=True
            )
            np.save(chunk_file, chunk_emb)
            print(f"Chunk {i} cached securely.")
            
        all_embeddings.append(chunk_emb)

    final_embeddings = np.vstack(all_embeddings)
    print(f"Embeddings matrix finalized. Shape: {final_embeddings.shape}")

    # 5. Build FAISS Index
    print("Building high-performance IndexFlatIP vector matrix...")
    faiss_embeddings = final_embeddings.astype('float32')
    faiss.normalize_L2(faiss_embeddings)
    
    index = faiss.IndexFlatIP(384)
    index.add(faiss_embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f"Vector index compiled and exported safely to {INDEX_PATH}.")

if __name__ == "__main__":
    run_indexing_pipeline()
