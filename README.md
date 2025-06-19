# Diabetes Prediction with Logistic Regression

This project predicts diabetes outcomes using logistic regression on the classic diabetes dataset. It features an interactive Streamlit app for data exploration, model training, and 3D visualization of predictions.

## Features
- Select features and target for prediction
- Adjust logistic regression hyperparameters
- Interactive 3D plot of predicted probabilities and decision surface
- Confusion matrix, accuracy, and ROC curve

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run scripts/streamlit_app.py
   ```
3. Open the app in your browser and explore the model and data.

## Project Structure
- `data/` — Contains the diabetes dataset (CSV)
- `scripts/` — Streamlit app and utilities
- `src/` — (Optional) Source code modules
- `tests/` — (Optional) Unit tests

## About
This project is for educational and demonstration purposes, showing how logistic regression can be used to predict diabetes and how to visualize model predictions interactively. 