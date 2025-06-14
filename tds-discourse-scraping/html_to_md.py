# /// script
# requires-python = "==3.12"
# dependencies = [
#     "html2text",
#     "rich",
#     "beautifulsoup4",
#     "pillow",
#     "google-genai",
#     "httpx",
# ]
# ///

import os
import json
import httpx
import html2text  # type: ignore
from bs4 import BeautifulSoup  # type: ignore
from PIL import Image
import tempfile
from google import genai

def get_image_description(image_url):
    """Download an image and get its description using Google GenAI."""
    client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

    response = httpx.get(image_url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    try:
        uploaded_file = client.files.upload(file=tmp_file_path)
        result = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[uploaded_file, "Caption this image"]
        )
        return result.text
    finally:
        os.remove(tmp_file_path)

def convert_all_chunks_to_md(chunks_path="chunks.json", output_dir="./markdowns"):
    os.makedirs(output_dir, exist_ok=True)

    with open(chunks_path, "r", encoding="utf-8") as chunks_file:
        for line in chunks_file:
            try:
                chunk = json.loads(line)
                html_content = chunk["content"]
                file_id = chunk["id"].replace("#", "_")
                output_path = os.path.join(output_dir, f"{file_id}.md")

                soup = BeautifulSoup(html_content, "html.parser")
                images = soup.find_all("img")

                for img in images:
                    src = img.get("src").replace("\\", "").strip('"')
                    description = get_image_description(src)

                    if img.parent.name == "a" and img.parent.parent and "lightbox-wrapper" in img.parent.parent.get("class", []):
                        img.parent.parent.replace_with(BeautifulSoup(f"<p>Image: {description}</p>", "html.parser"))
                    elif img.parent.name == "a":
                        img.parent.replace_with(BeautifulSoup(f"<p>Image: {description}</p>", "html.parser"))
                    else:
                        img.replace_with(BeautifulSoup(f"<p>Image: {description}</p>", "html.parser"))

                md_content = html2text.html2text(str(soup))
                cleaned_md = md_content.replace("\\n", "").replace("\n\n", "\n").strip()

                with open(output_path, "w", encoding="utf-8") as md_file:
                    md_file.write(cleaned_md)

            except Exception as e:
                print(f"Error processing chunk: {e}")

if __name__ == "__main__":
    convert_all_chunks_to_md()