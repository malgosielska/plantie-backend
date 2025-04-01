# Plant Disease Detection API

This is a FastAPI application for plant disease detection using machine learning. The backend leverages a Convolutional Neural Network (CNN) model to classify plant diseases based on images.

## Setup Instructions

### 1. Install **`uv`** (Python environment manager)
If you don’t have **`uv`** installed, you can install it using `pip`:
```bash
pip install uv
```

### 2. Create a Virtual Environment
Once uv is installed, set up a virtual environment for your project:
```bash
uv venv .venv
```
Activate the virtual environment:
On Linux/macOS:
```bash
source .venv/bin/activate
```
On Windows:
```bash
.\.venv\Scripts\activate
```

### 3. Install Dependencies
Now that the virtual environment is activated, install the project dependencies defined in your pyproject.toml file:
```bash
uv sync
```

### 4. Run the Application
Once dependencies are installed, you can run the FastAPI app using uvicorn:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Access the API Documentation
Once the server is running, you can access the auto-generated documentation at:

Swagger UI: http://127.0.0.1:8000/docs

ReDoc UI: http://127.0.0.1:8000/redoc

### 6. Ruff formatting
Before commiting your changes format your files with ruff:
```bash
ruff format .
```