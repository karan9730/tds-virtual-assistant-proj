# /// script
# requires-python = "==3.12"
# dependencies = [
#     "httpx",
#     "numpy",
# ]
# ///

import json
import os
import numpy as np  # type: ignore
import httpx  # type: ignore
from chunks_to_embeddings import embed  # type: ignore

# store collection course_collection.json in a variable
with open('course_collection.json', 'r') as file:
    course_collection = json.load(file)

def ask_question(question: str) -> list[dict]:
    """Find the most relevant chunks for a given question."""
    question_embedding = embed(question)

    results = []
    for chunk in course_collection['embeddings']:
        similarity = float(np.dot(question_embedding, chunk['embedding']) / (np.linalg.norm(question_embedding) * np.linalg.norm(chunk['embedding'])))
        results.append({
            'source': chunk['source'],
            'similarity': similarity,
            'text': chunk['text']
        })
    
    # Sort results by similarity in descending order
    results.sort(key=lambda x: x['similarity'], reverse=True)

    # save results to a json file
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    return results[:5]  # Return top 5 most relevant chunks

if __name__ == "__main__":
    question = "What is the purpose of base64?"
    results = ask_question(question)
    print(results)