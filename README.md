# Stock Prediction ML LSTM App

A Streamlit web application that predicts stock prices using a Long Short-Term Memory (LSTM) deep learning model, visualizes price trends with Moving Averages (50-day, 100-day, 200-day), and displays the latest stock news headlines.

## Features

- **Stock Price Prediction**: Predicts future stock prices based on historical trends using an LSTM model.
- **Interactive Moving Averages**: Compares stock closing prices with 50-day, 100-day, and 200-day Moving Averages.
- **Real-Time News**: Integrates NewsAPI to fetch the latest headlines for the selected stock ticker.

## Prerequisites

Before running the project, make sure you have:
1. **Python 3.8 - 3.11** installed on your system.
2. An active Internet connection to download historical stock data via `yfinance`.

## How to Run Locally

### 1. Clone or Open the Project Directory
Make sure your terminal is opened in the project's root folder:
```bash
cd c:\Users\sailk\Documents\projects\Stock-Prediction-ML-LSTM-
```

### 2. Set Up a Virtual Environment (Recommended)
Creating a virtual environment ensures dependencies don't conflict with other projects:
```bash
# Create the virtual environment
python -m venv venv

# Activate it:
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.\venv\Scripts\activate.bat
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Install all the required Python libraries using the generated `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
Launch the app locally by running:
```bash
streamlit run app.py
```

The app will compile and automatically open a tab in your web browser at `http://localhost:8501`.

---

## File Structure

- [app.py](file:///c:/Users/sailk/Documents/projects/Stock-Prediction-ML-LSTM-/app.py): The main Streamlit application script containing the frontend UI, plotting logic, and news aggregation.
- [Stock Predictions Model.keras](file:///c:/Users/sailk/Documents/projects/Stock-Prediction-ML-LSTM-/Stock%20Predictions%20Model.keras): The trained LSTM model loaded by the application.
- [requirements.txt](file:///c:/Users/sailk/Documents/projects/Stock-Prediction-ML-LSTM-/requirements.txt): List of dependencies.
- [LSTM model.ipynb](file:///c:/Users/sailk/Documents/projects/Stock-Prediction-ML-LSTM-/LSTM%20model.ipynb): Jupyter notebook containing the original data exploration and training pipeline.
