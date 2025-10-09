# AI-Stroke-Shield

[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28-orange?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

AI Stroke Shield is an intelligent healthcare project that leverages **artificial intelligence and machine learning** to detect early signs of stroke from medical data. The system assists healthcare professionals in timely diagnosis, improving patient outcomes and saving lives.

---

## 📁 Dataset Source

The dataset used in this project was sourced from:  

🔗 [Stroke Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data)

It includes health-related features such as **age, gender, hypertension, heart disease, BMI**, etc., and indicates whether a stroke occurred.

---

## 🧠 Model Description

The project uses a **machine learning classifier** trained on patient health data to predict the likelihood of a stroke. Key points:  

- **Algorithm**: Logistic Regression / Random Forest / [Specify your model]  
- **Input Features**: Age, Gender, Hypertension, Heart Disease, BMI, Smoking Status, etc.  
- **Output**: Probability of stroke (0–100%)  

The model is integrated with a **Streamlit web application** for easy interaction and visualization.

---

## ⚙️ Installation

Follow these steps to run AI-Stroke-Shield locally:

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/AI-Stroke-Shield.git
cd AI-Stroke-Shield

Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt

🚀 Usage

To start the Streamlit web app:

streamlit run chatbot/chatbot.py


Enter patient information in the input fields.

Click Predict to see the stroke risk probability.

View a progress bar and probability percentage for intuitive visualization.


✨ Features

Real-time stroke risk prediction

User-friendly web interface with Streamlit

Visual progress bar for probability

Easily extendable to include new features and models

🛠️ Technologies Used

Python 3.11

Streamlit

Scikit-learn / XGBoost / TensorFlow

Pandas & NumPy for data processing