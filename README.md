# Fraud Detection System with Streamlit

This project is a fraud detection system built with a deep learning model using TensorFlow, deployed with Streamlit for an interactive web interface. It allows users to input transaction details manually or upload a CSV file for fraud prediction. 

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
  - [Clone the Repository](#clone-the-repository)
  - [Install Dependencies](#install-dependencies)
  - [Download Models](#download-models)
- [Accessing the Application](#accessing-the-application)
- [File Structure](#file-structure)
- [Contributing](#contributing)

## Project Overview
This project aims to detect fraudulent transactions using a deep learning model and various transaction features. Users can either input transaction details manually or upload a CSV file containing transaction data for the fraud detection system to evaluate.  Data used- [IEEE-Fraud-detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

## Features
- **Manual Input**: Enter transaction details manually and get an immediate fraud prediction.
- **CSV Upload**: Upload a CSV file to analyze multiple transactions at once.
- **Prediction Results**: The system predicts whether the transaction is fraudulent or legitimate, displaying the probability.
- **Model**: The system uses an autoencoder for dimensionality reduction and a classifier to predict fraud.

## Technologies Used
- **TensorFlow**: Used for building and training the deep learning model.
- **Streamlit**: Used to create the interactive web app.
- **Scikit-learn**: Used for data preprocessing, including scaling and encoding.
- **Pandas**: Used for data manipulation and CSV file handling.
- **Imbalanced-learn**: Used for handling class imbalance with SMOTE.
- **Joblib**: Used for saving and loading models.

## Setup Instructions

### Clone the Repository
```bash
git clone https://github.com/your-username/fraud-detection-system.git
```
### Install Dependencies
Install the required dependencies listed in requirements.txt:
```bash
pip install -r requirements.txt
```
### Download Models
The trained models (encoder.keras, scaler.pkl, label_encoders.pkl) are stored in this repository.

## Accessing the Application
The application is deployed and accessible via the web. You can access it here:
Fraud Detection System App
No need to run the application locally—just visit the link to start interacting with the fraud detection system.

## File Structure
- `app.py`: The main Streamlit app file containing the frontend logic.
- `fraud_detection_model.py`: Contains the logic for loading the model and performing fraud detection.
- `requirements.txt`: Lists all the Python packages needed for the project.
- `Models` 
- `README.md`: This file.

## Contributing
We welcome contributions! Feel free to open an issue or create a pull request if you have improvements or bug fixes to share.







