# /// script
# requires-python = "==3.12"
# dependencies = [
#     "httpx",
#     "semantic-text-splitter",
#     "numpy",
#     "tqdm",
# ]
# ///

import os
import json
import time
import httpx
import numpy as np
from pathlib import Path
from semantic_text_splitter import MarkdownSplitter  # type: ignore
from tqdm import tqdm  # type: ignore


def get_embedding(text: str, max_retries: int = 5) -> list[float]:
    """Get OpenAI embedding using HTTPX."""
    api_key = os.environ.get("AIPIPE_TOKEN")
    
    url = "https://aipipe.org/openai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": text,
    }

    for attempt in range(max_retries):
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def get_chunks(file_path: str, chunk_size: int = 15000):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    splitter = MarkdownSplitter(chunk_size)
    return splitter.chunks(content)


if __name__ == "__main__":
    markdown_dir = Path("markdowns")
    files = list(markdown_dir.glob("*.md")) + list(markdown_dir.rglob("*.md"))

    all_chunks = []
    all_embeddings = []
    chunk_ids = []

    # Step 1: Load previous progress if it exists
    existing_file = Path("embeddings.npz")
    if existing_file.exists():
        print("Resuming from existing embeddings.npz...")
        existing = np.load(existing_file, allow_pickle=True)
        all_chunks = list(existing["chunks"])
        all_embeddings = list(existing["embeddings"])
        chunk_ids = list(existing["ids"])
        done_ids = set(chunk_ids)
    else:
        print("Starting fresh...")
        done_ids = set()

    # Step 2: Process new markdown files
    total_chunks = 0
    file_chunks = {}

    for file_path in files:
        chunks = get_chunks(file_path)
        file_chunks[file_path] = chunks
        total_chunks += len(chunks)

    print(f"Total .md files found: {len(files)}")
    print(f"Files skipped (0 chunks): {[str(f) for f, c in file_chunks.items() if len(c) == 0]}")
    print(f"Files that produced chunks: {len([f for f in file_chunks if len(file_chunks[f]) > 0])}")

    with tqdm(total=total_chunks, desc="Embedding chunks") as pbar:
        pbar.update(len(done_ids))  # Move progress bar ahead
        for file_path, chunks in file_chunks.items():
            file_stem = file_path.stem
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_stem}#{i + 1}"
                if chunk_id in done_ids:
                    continue  # Skip if already embedded
                try:
                    embedding = get_embedding(chunk)
                    all_chunks.append(chunk)
                    all_embeddings.append(embedding)
                    chunk_ids.append(chunk_id)
                    pbar.update(1)
                except Exception as e:
                    print(f"Skipping chunk from {file_path} due to error: {e}")
                    pbar.update(1)
                    continue

    # Step 3: Save everything back into the archive
    np.savez("embeddings.npz", chunks=all_chunks, embeddings=all_embeddings, ids=chunk_ids)
    print(f"✅ Saved {len(all_chunks)} embeddings to embeddings.npz")
