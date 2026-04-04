# 🩺 Anemia Detection System

## 📌 Overview
An AI-powered, non-invasive anemia screening system that analyzes **palpebral conjunctiva (inner eyelid) images** along with **user-reported symptoms** to predict anemia risk.

This system combines **deep learning + clinical indicators** to provide an accessible and quick screening tool.

---

## 🚨 Problem Statement
- Anemia often goes **undiagnosed** in early stages  
- Traditional diagnosis requires **invasive blood tests**  
- Lack of accessible screening tools for quick preliminary detection  

---

## 💡 Solution
This project uses:
- Eye images (conjunctiva region)
- Symptom inputs

to predict anemia using a trained deep learning model.

---

## ✨ Key Features
- 🔍 **Image-based anemia detection** using ResNet18
- 🧠 **Symptom-based scoring system**
- 🔗 **Fusion of image + symptom predictions**
- 🔥 **Grad-CAM visualization** for model interpretability
- 📸 **Supports camera capture & file upload**
- 📊 **Confidence bars for prediction clarity**
- 🖥️ **User-friendly GUI (Tkinter)**

---

## 🧠 Model Details
- Architecture: ResNet18
- Input: Cropped conjunctiva images
- Output: Binary classification (Anemia / Normal)
- Explainability: Grad-CAM heatmaps

---

## ⚙️ Tech Stack
- Python
- PyTorch
- OpenCV
- Tkinter
- NumPy
- PIL

---

## 🗂️ Project Structure

src/
│── frontend.py # GUI application
│── gradcam.py # Grad-CAM implementation
│── crop_utils.py # Eye region extraction
│── preprocess.py # Image preprocessing
│── prepare_dataset.py # Dataset preparation
│── augment_train_images.py # Data augmentation
│── clean_dataset.py # Data cleaning
│── train_model_2.py # Model training
│── Resnet+Normalization.py # Model architecture

------

## 📸 Usage Instructions
Select symptoms or "None of the Above"
Upload or capture 4 eye images
Follow these guidelines:
Ensure good lighting
Focus on inner eyelid (pink region)
Avoid blur and obstructions

------

## 📊 Output
Per-image prediction with confidence
Final aggregated prediction
Grad-CAM heatmap visualization
Risk percentage (Anemia vs Normal)

------

## ⚠️ Disclaimer

This is a screening tool only and not a medical diagnosis.
Consult a healthcare professional for accurate diagnosis.

------

## 🌱 Future Improvements
Mobile app integration
Real-time video analysis
Larger dataset training
Clinical validation

