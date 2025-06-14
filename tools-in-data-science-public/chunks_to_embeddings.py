# /// script
# requires-python = "==3.12"
# dependencies = [
#     "httpx",
#     "numpy",
# ]
# ///

import os
import json
import httpx  # type: ignore
import numpy as np # type: ignore

def embed(text: str) -> list[float]:
    """Get embedding vector for text using OpenAI API."""
    client =  httpx.Client()
    response = client.post(
        "https://api.jina.ai/v1/embeddings",
        headers = {'Content-Type': "application/json",'Authorization': f'Bearer {os.environ['JINA_AI_TOKEN']}'},
        json = {
            "model": "jina-clip-v2",
            "input": [{"text": text}]    
        },
        timeout=30
    )

    # write response to a json file
    with open("response.json", "w") as f:
        json.dump(response.json(), f, indent=4)
    return response.json()["data"][0]["embedding"]

with open('chunks.json', 'r') as f:
    chunks = f.readlines()

for i, chunk in enumerate(chunks):
    chunks[i] = json.loads(chunk)

# store chunk embeddings in a collection
collection = {
    "name": "course_content_chunks",
    "description": "Embeddings of chunks from course content",
    "embeddings": []
}

for chunk in chunks:
    embedding = embed(chunk["content"])
    collection["embeddings"].append({
        "source": chunk["id"],
        "embedding": embedding,
        "text": chunk["content"]
    })

# write collection to a json file
with open("course_collection.json", "w") as f:
    json.dump(collection, f, indent=4)