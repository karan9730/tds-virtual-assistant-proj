# TDS Virtual Assistant Project

## Overview
The TDS Virtual Assistant Project is a FastAPI application designed to process user input through a RESTful API. It provides endpoints for receiving questions and optional image data, with built-in error handling for common issues.

## Files
- **main.py**: Contains the FastAPI application with endpoints for processing input.
- **requirements.txt**: Lists the dependencies required for the project.

## Setup Instructions
1. **Clone the repository**:
   ```
   git clone https://github.com/karan9730/tds-virtual-assistant-proj
   cd tds-virtual-assistant-proj
   ```

2. **Create a virtual environment** (optional but recommended):
   ```
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```
   uvicorn main:app --reload
   ```

## Usage
- **GET /**: Returns a message indicating that the API is running.
- **POST /api/**: Accepts a JSON payload with the following structure:
  ```json
  {
      "question": "Your question here",
      "image": "Base64 encoded image string (optional)"
  }
  ```
  - The `question` field is required.
  - The `image` field is optional and should be a valid base64 encoded string if provided.

## Error Handling
- If the `question` field is missing, the API will return an error message.
- If the `image` field is provided but is not valid base64, an error message will be returned.

## License
This project is licensed under the MIT License.
