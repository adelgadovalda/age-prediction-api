"""
API REST para predicción de edad usando el modelo CNN entrenado
FastAPI + TensorFlow/Keras
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
from typing import Dict
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Age Prediction API",
    description="API para predecir edad a partir de imágenes faciales usando Deep Learning",
    version="1.0.0"
)

# Configurar CORS (para permitir peticiones desde frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constantes del modelo (deben coincidir con tu entrenamiento)
IMG_SIZE = 128
MODEL_PATH = "modelo_fotos.keras"

# Variable global para el modelo
model = None


def load_model():
    """Carga el modelo entrenado al iniciar la API"""
    global model
    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        logger.info(f"✅ Modelo cargado exitosamente desde {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Error al cargar el modelo: {str(e)}")
        raise


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesa la imagen para que sea compatible con el modelo
    
    Args:
        image: Imagen PIL
        
    Returns:
        Array numpy con la imagen preprocesada
    """
    # Convertir a RGB si es necesario
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionar a 128x128 (tamaño usado en entrenamiento)
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convertir a array numpy
    img_array = np.array(image)
    
    # Normalizar valores a [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Agregar dimensión de batch (modelo espera shape: (batch, height, width, channels))
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


@app.on_event("startup")
async def startup_event():
    """Evento que se ejecuta al iniciar la API"""
    load_model()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Endpoint raíz que muestra el frontend"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return JSONResponse({
            "message": "🎯 Age Prediction API - Deep Learning",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "predict": "/predict",
                "health": "/health",
                "docs": "/docs"
            }
        })


@app.get("/health")
async def health_check():
    """Verifica el estado de la API y el modelo"""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH
    }


@app.post("/predict")
async def predict_age(file: UploadFile = File(...)) -> Dict:
    """
    Predice la edad de una persona a partir de una imagen facial
    
    Args:
        file: Archivo de imagen (JPG, PNG, etc.)
        
    Returns:
        Diccionario con la edad predicha y metadatos
    """
    # Validar que el modelo esté cargado
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. El servidor necesita reiniciarse."
        )
    
    # Validar tipo de archivo
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo inválido: {file.content_type}. Se espera una imagen."
        )
    
    try:
        # Leer imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        logger.info(f"📸 Imagen recibida: {file.filename} | Tamaño: {image.size}")
        
        # Preprocesar imagen
        processed_image = preprocess_image(image)
        
        # Realizar predicción
        prediction = model.predict(processed_image, verbose=0)
        predicted_age = float(prediction[0][0])
        
        logger.info(f"🎯 Predicción: {predicted_age:.1f} años")
        
        # Redondear a 1 decimal
        predicted_age = round(predicted_age, 1)
        
        return {
            "success": True,
            "predicted_age": predicted_age,
            "predicted_age_rounded": round(predicted_age),
            "filename": file.filename,
            "image_size": image.size,
            "message": f"La edad estimada es {round(predicted_age)} años"
        }
        
    except Exception as e:
        logger.error(f"❌ Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la imagen: {str(e)}"
        )


@app.post("/predict_batch")
async def predict_age_batch(files: list[UploadFile] = File(...)) -> Dict:
    """
    Predice la edad para múltiples imágenes
    
    Args:
        files: Lista de archivos de imagen
        
    Returns:
        Diccionario con las predicciones para cada imagen
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible."
        )
    
    results = []
    
    for file in files:
        try:
            # Validar tipo
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Tipo de archivo inválido"
                })
                continue
            
            # Procesar imagen
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            processed_image = preprocess_image(image)
            
            # Predicción
            prediction = model.predict(processed_image, verbose=0)
            predicted_age = round(float(prediction[0][0]), 1)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "predicted_age": predicted_age,
                "predicted_age_rounded": round(predicted_age)
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total_images": len(files),
        "results": results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
