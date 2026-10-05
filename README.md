# ❤️ Heart Disease Prediction using Machine Learning

## Project Overview

This project demonstrates how Machine Learning can be used to classify heart-disease risk from selected health-related features.

The project uses a **Random Forest Classifier** and provides an interactive **Streamlit** interface for educational demonstration.

> **Medical disclaimer:** This project is for educational purposes only. It is not a medical diagnostic system and should not be used to make healthcare decisions.

## Features

- Heart disease classification
- Random Forest Machine Learning model
- Interactive Streamlit interface
- Model probability output
- Accuracy evaluation
- Beginner-friendly Python project structure
- Ready for GitHub

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- GitHub

## Project Structure

```text
Heart_Disease_ML/
│
├── app/
│   └── app.py
├── data/
│   └── heart_disease.csv
├── models/
│   ├── heart_disease_model.pkl
│   └── metrics.txt
├── src/
│   └── train_model.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Machine Learning Workflow

```text
Dataset
   ↓
Data Preparation
   ↓
Train-Test Split
   ↓
Random Forest Classification
   ↓
Model Evaluation
   ↓
Prediction
   ↓
Streamlit Application
```

## Input Features

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Maximum heart rate
- Exercise-induced angina
- ST depression (oldpeak)
- ST slope

## Target

`heart_disease`

- `0` = lower-risk class in this educational dataset
- `1` = higher-risk class in this educational dataset

## Model Performance

Accuracy on the included test split: **0.8417**

The dataset included in this repository is **synthetic and created for educational demonstration**.

## Installation

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python src/train_model.py
```

## Run the Application

```bash
streamlit run app/app.py
```

## Future Improvements

- Use a validated real-world public dataset
- Add cross-validation
- Compare multiple classification algorithms
- Add confusion matrix and ROC-AUC visualization
- Add model explainability
- Improve input validation and accessibility

## Author

**Vanshika Agarwal**

B.Tech CSE (AI/ML)
