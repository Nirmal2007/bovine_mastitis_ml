# Mastitis Monitoring System

Real-time bovine telemetry dashboard with Machine Learning inference for early mastitis detection.

## Prerequisites

Make sure you have **Python 3.8 or higher** installed on your system.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/mastitis-monitoring-system.git
   cd mastitis-monitoring-system
   ```

2. **Create and activate a virtual environment (Recommended):**
   * **Windows:**
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   * **Mac / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install required Python libraries:**
   Install all necessary dependencies using `pip`:
   ```bash
   pip install pandas scikit-learn joblib flask flask-socketio pyserial
   ```

## Running the Project

1. **Run the Flask Web Dashboard:**
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to: `http://localhost:5000`

3. **Run the ML Model Inference Showcase:**
   ```bash
   python predict_inference.py
