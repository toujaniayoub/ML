# 📱 SmartPredict – Machine Learning Web Application

![SmartPredict Banner](assets/img//testimonials/logo.png)

> 🔮 *SmartPredict is a machine learning-powered web application that helps users select smartphones based on their personalized needs and predicts smartphone prices using advanced algorithms.*  

---

## 📖 Table of Contents
- [✨ Features](#-features)
- [🧠 Machine Learning Models](#-machine-learning-models)
- [🛠 Tech Stack](#-tech-stack)
- [📂 Project Structure](#-project-structure)
- [⚙️ Installation & Setup](#️-installation--setup)
- [📸 Screenshots](#-screenshots)
- [📊 Demo](#-demo)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## ✨ Features
✅ Predicts smartphone prices based on technical features.  
✅ Recommends smartphones tailored to user preferences.  
✅ User-friendly **Flask web application**.  
✅ Interactive visualizations powered by **Power BI**.  
✅ Responsive design with **Bootstrap, HTML, CSS, JavaScript**.  

---

## 🧠 Machine Learning Models
The following ML algorithms were trained and evaluated for price prediction:
- **K-Nearest Neighbors (KNN)**
- **Random Forest**
- **XGBoost**

📊 Best performing model: **XGBoost (highest accuracy on test set).**

---

## 🛠 Tech Stack

**Frontend:**  
- HTML, CSS, JavaScript, Bootstrap  

**Backend:**  
- Flask (Python)  

**Machine Learning:**  
- Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, XGBoost  

**Data Visualization:**  
- Power BI  

---

## 📂 Project Structure
```bash
SmartPredict/
│
├── app/                   # Flask application
│   ├── static/            # CSS, JS, images
│   ├── templates/         # HTML files
│   └── routes.py          # Main Flask routes
│
├── models/                # Trained ML models (Pickle files)
├── notebooks/             # Jupyter Notebooks for data exploration
├── data/                  # Raw and cleaned datasets
├── requirements.txt       # Dependencies
├── app.py                 # Main Flask entry point
└── README.md              # Project documentation
