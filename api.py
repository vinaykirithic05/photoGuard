"""
=========================================================
PhotoGuard AI
FastAPI Inference & Detection Server

Author : Vinay Kirithic

Description:
REST API Backend serving PhotoGuard CNN predictions and Grad-CAM
explainability heatmaps for the Flutter mobile application.
=========================================================
"""

import os
import base64
from io import BytesIO
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from modules.cnn.predict import PhotoGuardPredictor
from modules.cnn.gradcam import generate_gradcam_overlay
from modules.paths import OUTPUT_DIR

app = FastAPI(
    title="PhotoGuard AI API",
    description="Backend service for AI Image Detection & Explainability",
    version="1.0.0"
)

# Enable CORS for Flutter Web & Mobile requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Predictor Engine
predictor = PhotoGuardPredictor()

# Static directory for serving Grad-CAM images
GRADCAM_SERVE_DIR = OUTPUT_DIR / "gradcam"
GRADCAM_SERVE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/gradcam", StaticFiles(directory=str(GRADCAM_SERVE_DIR)), name="gradcam")


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "PhotoGuard AI Inference Server",
        "version": "1.0.0"
    }


@app.post("/api/v1/predict")
async def predict_photo(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file and returns:
    - Prediction label ('AI' vs 'Real')
    - Confidence percentage
    - Class probability scores
    - Grad-CAM heatmap base64 string
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (.jpg, .jpeg, .png, .webp)")

    # Save uploaded file temporarily inside outputs/temp
    temp_dir = OUTPUT_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename

    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # Run CNN Inference
        result = predictor.predict_image(temp_path)

        # Generate Grad-CAM Heatmap
        gradcam_filename = f"gradcam_{Path(file.filename).stem}.jpg"
        gradcam_output_path = generate_gradcam_overlay(temp_path, output_filename=gradcam_filename)

        # Encode Grad-CAM overlay image to Base64 for instant Flutter display
        with open(gradcam_output_path, "rb") as image_file:
            encoded_gradcam = base64.b64encode(image_file.read()).decode("utf-8")

        result["gradcam_base64"] = encoded_gradcam
        result["gradcam_url"] = f"/static/gradcam/{gradcam_filename}"

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

    finally:
        # Clean temp uploaded file
        if temp_path.exists():
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
