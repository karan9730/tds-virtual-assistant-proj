# /// script
# requires-python = "==3.12"
# dependencies = [
#     "argparse",
#     "fastapi",
#     "httpx",
#     "markdownify",
#     "numpy",
#     "semantic-text-splitter",
#     "tqdm",
#     "uvicorn",
#     "google-genai",
#     "pillow",
# ]
# ///


import base64
import json
import tempfile
import time
import numpy as np # type: ignore
import os
import re
from fastapi import FastAPI # type: ignore
from pydantic import BaseModel # type: ignore
import httpx # type: ignore
import html
from google import genai
from google.genai.types import GenerateContentConfig # type: ignore
from fastapi.responses import JSONResponse # type: ignore

app = FastAPI()

class InputData(BaseModel):
    question: str
    image: str = None

def get_embedding(text: str, max_retries: int = 3) -> list[float]:
    """Get embedding for text chunk using AIPipe's OpenAI-compatible endpoint with retry logic."""

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
            if attempt == max_retries - 1:
                print(f"❌ Failed to get embedding after {max_retries} attempts: {e}")
                raise
            else:
                wait_time = 2 ** attempt
                print(f"⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    raise Exception("Max retries exceeded")

def get_image_description(image_input: str) -> str:
    """
    Accepts either a URL or a base64-encoded image string (with or without the data URI prefix).
    Downloads or decodes the image, uploads it to Gemini, and returns a description.
    """
    client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

    # Detect if input is a base64 image
    is_base64 = image_input.startswith("data:image/") or not image_input.startswith("http")

    if is_base64:
        # Remove data URI prefix if present
        if image_input.startswith("data:image/"):
            image_input = image_input.split(",")[1]

        image_bytes = base64.b64decode(image_input)
    else:
        # Otherwise, treat as URL and download the image
        response = httpx.get(image_input)
        response.raise_for_status()
        image_bytes = response.content

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(image_bytes)
        tmp_file_path = tmp_file.name

    try:
        uploaded_file = client.files.upload(file=tmp_file_path)
        result = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[uploaded_file, "Describe the content of this image in detail, focusing on any text, objects, or relevant features that could help answer questions about it."]
        )
        return result.text
    finally:
        os.remove(tmp_file_path)

def load_embeddings():
    """Load ids, chunks and embeddings from npz file"""
    data = np.load("tds-discourse-scraping/embeddings.npz", allow_pickle=True)
    return data["ids"], data["chunks"], data["embeddings"]

def generate_llm_response(question: str, context: str, max_retries: int = 3):
    client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

    prompt = (
        "You are a helpful teaching assistant. "
        "Answer the user's question using the provided context. "
        "Quote relevant parts directly (without escaping quotes). "
        "Format using plain text (no HTML, no backslashes).\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=GenerateContentConfig(
                    max_output_tokens=512,
                    temperature=0.3,
                    top_p=0.9,
                    top_k=40
                )
            )
            # ✅ Extract raw text properly from the response
            return response.candidates[0].content.parts[0].text
        except Exception as e:
            if "overloaded" in str(e).lower() or "503" in str(e):
                wait_time = 2 ** attempt
                print(f"⚠️ Gemini overloaded. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            elif attempt == max_retries - 1:
                raise
            else:
                print(f"⚠️ Error on attempt {attempt+1}: {e}")
                time.sleep(1)

def clean_response(text: str) -> str:
    """Clean Gemini output and strip redundant wrapping quotes and artifacts."""
    if not text:
        return ""

    text = text.strip()

    # Strip surrounding quotes if present
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()

    # Unescape HTML
    text = html.unescape(text)

    # Remove only escaped double quotes: \" → "
    text = text.replace(r'\"', '"')
    text = text.replace(r'\n\n', '\n')
    text = text.replace(r'\n', ' ')
    text = text.replace(r'##', '')

    # Optional: remove zero-width spaces or fix newlines
    text = text.replace("\u200b", "")
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text

def build_reference_links(ids: list, chunks: list, max_links: int = 2):
    from pathlib import Path

    with open("tds-discourse-scraping/discourse_topic_list.json", "r") as f:
        topic_map = {str(item["id"]): item["slug"] for item in map(json.loads, f)}

    links = []

    for i, id_str in enumerate(ids[:max_links]):
        chunk = chunks[i]
        base_id = id_str.split("#")[0]

        # Generate URL
        if "_" in base_id:
            topic_id, post_number = base_id.split("_")
            slug = topic_map.get(topic_id)
            if slug:
                url = f"https://discourse.onlinedegree.iitm.ac.in/t/{slug}/{topic_id}/{post_number}"
            else:
                url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}/{post_number}"
        else:
            url = f"https://tds.s-anand.net/#/{base_id}"

        # Clean snippet text (limit to 250 characters from full chunk)
        cleaned_text = clean_response(chunk.strip())
        snippet = cleaned_text[:250].rstrip() + "..." if len(cleaned_text) > 250 else cleaned_text

        links.append({
            "url": url,
            "text": snippet
        })

    return links

def answer(question: str, image: str = None):
    loaded_ids, loaded_chunks, loaded_embeddings = load_embeddings()

    if image:
        image_description = get_image_description(f"data:image/png;base64,{image}")
        question += f"\n\nImage description:\n{image_description}"

    question_embedding = get_embedding(question)

    similarities = np.dot(loaded_embeddings, question_embedding) / (
        np.linalg.norm(loaded_embeddings, axis=1) * np.linalg.norm(question_embedding)
    )

    top_indices = np.argsort(similarities)[-5:][::-1]
    top_chunks = [loaded_chunks[i] for i in top_indices]
    top_ids = [loaded_ids[i] for i in top_indices]

    llm_response = generate_llm_response(question, "\n".join(top_chunks))

    if llm_response is None:
        return {"error": "No response from Gemini. Please try again."}

    clean_answer = clean_response(llm_response)
    references = build_reference_links(top_ids, top_chunks)

    return {
        "answer": clean_answer,
        "links": references
    }

def is_valid_base64_image(b64_string):
    try:
        # Try to decode the string
        decoded = base64.b64decode(b64_string, validate=True)
        return True
    except Exception:
        return False

@app.get("/")
def default():
    return {"message": "Api is running"}

@app.post("/api/")
async def api_answer(data: InputData):
    if data.image:
        if not is_valid_base64_image(data.image):
            return JSONResponse({"error": "Invalid base64 image format"})

    try:
        result = answer(data.question, data.image)
        return JSONResponse({
            "answer": result["answer"].replace('"', "'"),
            "links": result.get("links", [])
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



if __name__ == "__main__":
    import uvicorn # type: ignore
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)  # Default port for FastAPI
    # main()
