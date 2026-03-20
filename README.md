# 🧠 NeuroVision AI

### AI-Powered Brain Tumor Detection System with Explainable AI

---

## 🚀 Overview

NeuroVision AI is a deep learning-based medical imaging system designed to classify brain MRI scans into four categories:

* Glioma
* Meningioma
* Pituitary Tumor
* No Tumor

The system leverages **EfficientNetB0 with transfer learning** and integrates **Grad-CAM visualization** to provide explainable predictions.

---

## 🎯 Key Features

* ✅ MRI Image Classification (4 Classes)
* ✅ Achieves ~87% Test Accuracy
* ✅ Grad-CAM Visualization for Explainability
* ✅ Streamlit-based Interactive UI
* ✅ Prediction Confidence Scores
* ✅ Confusion Matrix & Performance Evaluation
* ✅ Downloadable Report (optional extension)

---

## 🧠 Model Details

* **Architecture**: EfficientNetB0 (Transfer Learning)
* **Input Size**: 224 × 224
* **Training Strategy**:

  * Data Augmentation
  * Feature Extraction
  * Fine-tuning (last layers)
* **Optimizer**: Adam
* **Loss Function**: Categorical Crossentropy

---

## 📊 Performance

* **Training Accuracy**: ~90%
* **Validation Accuracy**: ~90%
* **Test Accuracy**: ~87%

### 🔍 Observations:

* Strong performance on Glioma & Pituitary classes
* Minor confusion between Meningioma and No Tumor

---

## 🔥 Explainable AI (Grad-CAM)

Grad-CAM is used to visualize model attention and highlight important regions in MRI scans.

➡️ This helps validate predictions and improves model transparency.

---

## 🖥️ Project Structure

```
brain-tumor-ai/
│
├── app.py                 # Streamlit UI
├── gradcam.py             # Grad-CAM implementation
├── utils.py               # Helper functions
│
├── model/
│   ├── final_model.keras
│   ├── class_indices.json
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/lavanya-sathasivam/neurovision-ai.git
cd neurovision-ai

python -m venv venv
venv\Scripts\activate   # (Windows)

pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📸 Sample Output

* MRI Image Input
* Predicted Tumor Type
* Confidence Score
* Grad-CAM Heatmap Visualization

---

## 📌 Future Improvements

* Integration with larger medical datasets
* Multi-modal imaging support (X-RAY + CT + MRI)
* Model deployment on cloud
* Real-time clinical integration

---

## 💡 Author

Lavanya Satha
AIML Student | Aspiring AI Engineer

---

## ⭐ If you found this useful, consider giving a star!
