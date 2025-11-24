# 🎯 Age Prediction API - Deep Learning Project

**Proyecto de Procesamiento de Imágenes con Deep Learning**  
Módulo: IDA803-9792-225081-ONL-PROCESAMIENTO DE IMÁGENES

API REST para predicción de edad a partir de imágenes faciales usando una Red Neuronal Convolucional (CNN) entrenada con TensorFlow/Keras.

---

## 🚀 Características

- Predicción de edad mediante CNN
- API REST con FastAPI
- Frontend interactivo con HTML/CSS/JavaScript
- Dockerizado para fácil deployment
- Modelo entrenado con 9,082 imágenes

---

## 📋 Requisitos

- Python 3.10+
- TensorFlow 2.18.0
- FastAPI
- Docker (para deployment)

---

## 🛠️ Instalación Local

```bash
# Clonar repositorio
git clone <tu-repo>
cd <tu-repo>

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servidor
uvicorn main:app --reload
```

La API estará disponible en: http://localhost:8000

---

## 🐳 Docker

```bash
# Construir imagen
docker build -t age-prediction-api .

# Ejecutar contenedor
docker run -p 8000:8000 age-prediction-api
```

---

## 📡 Endpoints

- `GET /` - Información de la API
- `GET /health` - Health check
- `POST /predict` - Predecir edad de una imagen
- `POST /predict_batch` - Predecir edad de múltiples imágenes
- `GET /docs` - Documentación interactiva (Swagger)

---

## 🎨 Frontend

Abrir `index.html` en el navegador para acceder a la interfaz gráfica.

---

## 📊 Modelo

- **Arquitectura:** CNN con 4 bloques convolucionales
- **Input:** Imágenes 128x128 RGB
- **Output:** Edad estimada (regresión)
- **Dataset:** 17,058 imágenes (train + test)
- **Rango de edad:** 1-100 años

---

## 👨‍💻 Autor

Proyecto desarrollado para el curso de Procesamiento de Imágenes

---

## 📄 Licencia

MIT License
