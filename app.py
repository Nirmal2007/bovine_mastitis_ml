import base64
import io
import json
import os
import sqlite3
import sys
import threading
import time
from flask import Flask, jsonify, render_template_string, request
from gtts import gTTS
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from pydub import AudioSegment
import requests

# Import YOLO (Ultralytics)
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. YOLO features will be disabled.")

app = Flask(__name__)

# --- GLOBAL VARIABLES & CONFIGURATION ---
DB_NAME = "cow_farm.db"
MODEL_FILE = "mastitis_risk_model.pkl"
ENCODER_FILE = "mastitis_encoder.pkl"
DATASET_CSV = "cows_dataset.csv"

# Update this path if your weights are located elsewhere
MODEL_PATH_YOLO = r"runs\classify\train\weights\best.pt"

# Wi-Fi ESP32-CAM IPs & Endpoints
FIXED_TERMINAL_IP = "10.97.0.246"
SENSOR_URL = f"http://{FIXED_TERMINAL_IP}/api/sensors"
CAMERA_URL = f"http://{FIXED_TERMINAL_IP}/camera"
PCM_STREAM_URL = f"http://{FIXED_TERMINAL_IP}/api/play_pcm"

# --- GLOBAL STATE ---
data_lock = threading.Lock()

latest_sensor_data = {
    "body_temp": 0.0,
    "ambient_temp": 0.0,
    "humidity": 0.0,
    "activity": 0.0,
    "activity_state": "WAITING",
    "milk_yield": 0.0,
    "milk_ec": 0.0,
    "last_seen": 0,
}

current_selected_cow_id = 1
esp_connected_status = False

# Camera & YOLO State
latest_camera_base64 = ""
camera_ok = False
camera_last_seen = 0
farm_cleanliness_status = "Yes"  # Default to clean
yolo_confidence = 0.0

# Risk State
current_risk_percentage = "Gathering 7 Days Data..."
current_risk_status = "WAITING"
current_precaution = "Waiting for AI result"
model_risk_value = 0.0


# --- TEXT-TO-SPEECH STREAMER ---
def speak_to_esp32(text_to_speak):
    """Generates speech via gTTS and sends raw PCM audio to the ESP32 speaker."""
    try:
        print(f"[TTS] Generating speech: '{text_to_speak}'")

        # 1. Synthesize Speech MP3 via Google TTS
        tts = gTTS(text=text_to_speak, lang="en")
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        # 2. Extract Raw Bytes & Send Stream
        requests.post(
            PCM_STREAM_URL,
            data=mp3_fp.read(),
            headers={"Content-Type": "audio/mpeg"},
            timeout=10,
        )
        print("[TTS] Voice stream sent to ESP32 speaker successfully.")
    except Exception as e:
        print(f"[TTS ERROR] Failed to send audio stream: {e}")


# --- 1. INITIALIZATION & DATASET GENERATION ---
def init_db_and_csv():
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA journal_mode=WAL;")
    c = conn.cursor()
    c.execute("""
              CREATE TABLE IF NOT EXISTS daily_activity
              (
                  day_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  cow_id INTEGER,
                  env_temp REAL,
                  env_humid REAL,
                  body_temp REAL,
                  cows_mov REAL,
                  milk_condu REAL,
                  milk_yield REAL
              )
              """)
    conn.commit()
    conn.close()

    if not os.path.exists(DATASET_CSV):
        print(f"[INFO] {DATASET_CSV} missing. Generating default dataset...")
        df = pd.DataFrame({
            "Cow_ID": [1, 2, 3],
            "Age": [8, 4, 6],
            "Breed": ["Jersey", "Holstein", "Guernsey"],
            "Farm_Clea": ["No", "Yes", "Yes"],
        })
        df.to_csv(DATASET_CSV, index=False)


def load_assets():
    print("\n==================================================")
    print(" 🛠️ INITIALIZING SYSTEM & ML ASSETS")
    print("==================================================")
    try:
        loaded_model = joblib.load(MODEL_FILE)
        loaded_encoder = joblib.load(ENCODER_FILE)
        print("[ML STATUS] ✅ Tabular Model successfully loaded")
    except FileNotFoundError:
        print(f"[ML ERROR] ❌ Missing '{MODEL_FILE}' or '{ENCODER_FILE}'!")
        sys.exit(1)

    y_model = None
    if YOLO_AVAILABLE and os.path.exists(MODEL_PATH_YOLO):
        try:
            y_model = YOLO(MODEL_PATH_YOLO)
            print("[YOLO STATUS] ✅ Image classification model loaded")
        except Exception as e:
            print(f"[YOLO ERROR] Could not load YOLO model: {e}")
    else:
        print(
            f"[YOLO WARNING] YOLO model file not found at {MODEL_PATH_YOLO}"
        )

    print("==================================================\n")
    return loaded_model, loaded_encoder, y_model


init_db_and_csv()
ml_model, ml_encoder, yolo_model = load_assets()


def get_cow_profile(cow_id):
    try:
        df = pd.read_csv(DATASET_CSV)
        cow_row = df[df["Cow_ID"] == int(cow_id)]
        if not cow_row.empty:
            return {
                "Age": float(cow_row.iloc[0]["Age"]),
                "Breed": str(cow_row.iloc[0]["Breed"]),
                "Farm_Cleanliness": str(
                    cow_row.iloc[0].get("Farm_Clea", "Yes")
                ),
            }
    except Exception:
        pass
    return {"Age": 4.0, "Breed": "Holstein", "Farm_Cleanliness": "Yes"}


def determine_precaution(status):
    status = str(status).upper()
    if status == "CRITICAL":
        return "Isolate cow immediately. Consult a veterinarian."
    elif status == "WATCH":
        return "Monitor closely. Maintain hygiene."
    elif status == "HEALTHY":
        return "Normal behavior patterns. Continue routine."
    return "Waiting for sufficient data."


# --- 2. CAMERA & YOLO READER THREAD ---
def read_camera():
    global latest_camera_base64, camera_ok, camera_last_seen
    global farm_cleanliness_status, yolo_confidence

    while True:
        try:
            response = requests.get(CAMERA_URL, timeout=5)
            if response.status_code == 200:
                image_bytes = response.content
                if len(image_bytes) > 100:
                    encoded = base64.b64encode(image_bytes).decode("utf-8")

                    if yolo_model is not None:
                        try:
                            img = Image.open(io.BytesIO(image_bytes))
                            results = yolo_model.predict(
                                source=img, verbose=False
                            )

                            class_names = results[0].names
                            unclean_class_id = None
                            for idx, name in class_names.items():
                                if name.lower() == "unclean":
                                    unclean_class_id = idx
                                    break

                            if unclean_class_id is not None:
                                probs = results[0].probs.data.tolist()
                                unclean_conf = probs[unclean_class_id]

                                with data_lock:
                                    yolo_confidence = unclean_conf * 100
                                    farm_cleanliness_status = (
                                        "No" if unclean_conf >= 0.70 else "Yes"
                                    )
                        except Exception as e:
                            print("[YOLO PREDICT ERROR]", e)

                    with data_lock:
                        latest_camera_base64 = encoded
                        camera_ok = True
                        camera_last_seen = time.time()
                else:
                    camera_ok = False
            else:
                camera_ok = False
        except Exception:
            camera_ok = False
        time.sleep(3)


# --- 3. SENSOR READER THREAD ---
def read_sensors_wifi():
    global latest_sensor_data, esp_connected_status
    print(f"🔌 [WIFI STATUS] Fetching sensors from {SENSOR_URL}...")

    while True:
        try:
            response = requests.get(SENSOR_URL, timeout=3)
            if response.status_code == 200:
                data = response.json()
                with data_lock:
                    latest_sensor_data.update({
                        "body_temp": float(data.get("body_temp", 0)),
                        "ambient_temp": float(data.get("ambient_temp", 0)),
                        "humidity": float(data.get("humidity", 0)),
                        "activity": float(data.get("activity", 0)),
                        "activity_state": data.get("activity_state", "NA"),
                        "milk_yield": float(data.get("milk_yield", 0)),
                        "milk_ec": float(data.get("milk_ec", 0)),
                        "last_seen": time.time(),
                    })
                esp_connected_status = True
            else:
                esp_connected_status = False
        except Exception:
            esp_connected_status = False
        time.sleep(2.5)


# --- 4. PROTOTYPE "DAY" LOOP & ML INFERENCE ---
def record_daily_data():
    global current_risk_percentage, current_risk_status, current_precaution

    while True:
        time.sleep(2.5)

        try:
            with data_lock:
                if (
                    latest_sensor_data["last_seen"] == 0
                    or (time.time() - latest_sensor_data["last_seen"]) > 5.0
                ):
                    continue

                calculated_yield = (
                    latest_sensor_data["milk_yield"] / 1.032
                    if latest_sensor_data["milk_yield"] > 0
                    else 0
                )
                b_temp, a_temp = (
                    latest_sensor_data["body_temp"],
                    latest_sensor_data["ambient_temp"],
                )
                hum, act, ec = (
                    latest_sensor_data["humidity"],
                    latest_sensor_data["activity"],
                    latest_sensor_data["milk_ec"],
                )

            conn = sqlite3.connect(DB_NAME, timeout=10)
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO daily_activity (cow_id, env_temp, env_humid, body_temp, cows_mov, milk_condu, milk_yield)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(current_selected_cow_id),
                    a_temp,
                    hum,
                    b_temp,
                    act,
                    ec,
                    calculated_yield,
                ),
            )
            conn.commit()

            c.execute(
                """
                SELECT env_temp, env_humid, body_temp, cows_mov, milk_condu, milk_yield
                FROM daily_activity
                WHERE cow_id = ?
                ORDER BY day_id DESC LIMIT 7
                """,
                (int(current_selected_cow_id),),
            )
            rows = c.fetchall()
            conn.close()

            if len(rows) == 7:
                rows.reverse()
                profile = get_cow_profile(current_selected_cow_id)

                with data_lock:
                    current_cleanliness = farm_cleanliness_status

                new_cow_data = {
                    "Age": profile["Age"],
                    "Breed": profile["Breed"],
                    "Farm_Cleanliness": current_cleanliness,
                }

                for i, row in enumerate(rows):
                    day_num = i + 1
                    new_cow_data[f"env_temp_day{day_num}"] = row[0]
                    new_cow_data[f"env_humidity_day{day_num}"] = row[1]
                    new_cow_data[f"body_temp_day{day_num}"] = row[2]
                    new_cow_data[f"cows_movement_day{day_num}"] = row[3]
                    new_cow_data[f"milk_conductivity_day{day_num}"] = row[4]
                    new_cow_data[f"milk_yield_day{day_num}"] = row[5]

                df_new = pd.DataFrame([new_cow_data])
                categorical_cols = ["Breed", "Farm_Cleanliness"]
                df_new[categorical_cols] = ml_encoder.transform(
                    df_new[categorical_cols]
                )

                # ML Prediction
                model_risk = float(
                    np.clip(ml_model.predict(df_new)[0], 0, 100)
                )
                camera_adjustment = (
                    15.0 if current_cleanliness == "No" else 0.0
                )
                if not camera_ok:
                    camera_adjustment = 5.0

                final_risk = float(
                    np.clip(model_risk + camera_adjustment, 0, 100)
                )
                current_risk_percentage = f"{final_risk:.2f}%"

                if final_risk >= 75:
                    current_risk_status = "CRITICAL"
                elif final_risk >= 40:
                    current_risk_status = "WATCH"
                else:
                    current_risk_status = "HEALTHY"

                current_precaution = determine_precaution(current_risk_status)

                # Auto-stream spoken audio for Critical Status
                if current_risk_status == "CRITICAL":
                    spoken_msg = f"Attention farmer. Critical mastitis risk level at {current_risk_percentage}. {current_precaution}"
                    threading.Thread(
                        target=speak_to_esp32, args=(spoken_msg,), daemon=True
                    ).start()

            else:
                current_risk_percentage = (
                    f"Gathering Data ({len(rows)}/7 days)..."
                )
                current_risk_status = "WAITING"
                current_precaution = "Waiting for AI result"

        except Exception as e:
            print(f"[DB ERROR IN LOOP]: {e}")


# Start Background Threads
threading.Thread(target=read_camera, daemon=True).start()
threading.Thread(target=read_sensors_wifi, daemon=True).start()
threading.Thread(target=record_daily_data, daemon=True).start()

# --- 5. EMBEDDED LOGBOOK HTML DASHBOARD ---
def get_html_page():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Template index.html not found</h1>"

HTML_PAGE = get_html_page()


# --- 6. FLASK ROUTES ---
# --- 6. FLASK ROUTES ---
@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception:
        return render_template_string(HTML_PAGE)


@app.route("/api/history", methods=["GET"])
def api_history():
    try:
        conn = sqlite3.connect(DB_NAME, timeout=5)
        c = conn.cursor()
        c.execute("""
            SELECT day_id, timestamp, env_temp, env_humid, body_temp, cows_mov, milk_condu, milk_yield
            FROM daily_activity
            WHERE cow_id = ?
            ORDER BY day_id DESC LIMIT 15
        """, (int(current_selected_cow_id),))
        rows = c.fetchall()
        conn.close()
        rows.reverse()
        history = []
        for r in rows:
            history.append({
                "day_id": r[0],
                "timestamp": r[1],
                "env_temp": float(r[2] or 0),
                "env_humid": float(r[3] or 0),
                "body_temp": float(r[4] or 0),
                "cows_mov": float(r[5] or 0),
                "milk_condu": float(r[6] or 0),
                "milk_yield": float(r[7] or 0)
            })
        return jsonify({"history": history, "selected": current_selected_cow_id})
    except Exception as e:
        return jsonify({"history": [], "error": str(e)})


@app.route("/api/cows", methods=["GET"])
def get_cows():
    cow_ids = [1]
    if os.path.exists(DATASET_CSV):
        try:
            df = pd.read_csv(DATASET_CSV)
            if "Cow_ID" in df.columns:
                cow_ids = df["Cow_ID"].astype(int).unique().tolist()
        except Exception:
            pass
    return jsonify({"cows": cow_ids, "selected": current_selected_cow_id})


@app.route("/api/select_cow", methods=["POST"])
def select_cow():
    global current_selected_cow_id, current_risk_percentage, current_risk_status, current_precaution
    data = request.get_json()
    if data and "cow_id" in data:
        current_selected_cow_id = int(data["cow_id"])
        current_risk_percentage = "Switching Cow... Gathering Data..."
        current_risk_status = "WAITING"
        current_precaution = "Waiting for AI result"
        return jsonify(
            {"status": "success", "selected": current_selected_cow_id}
        )
    return jsonify({"status": "error"}), 400


@app.route("/api/data", methods=["GET"])
def api_data():
    with data_lock:
        connected = esp_connected_status and (
            (time.time() - latest_sensor_data["last_seen"]) < 10
        )
        cam_avail = camera_ok and ((time.time() - camera_last_seen) < 10)

        return jsonify({
            "sensors": latest_sensor_data,
            "esp_connected": connected,
            "camera_available": cam_avail,
            "camera_image": latest_camera_base64,
            "farm_cleanliness": farm_cleanliness_status,
            "yolo_confidence": yolo_confidence,
            "risk_percentage": current_risk_percentage,
            "risk_status": current_risk_status,
            "precaution": current_precaution,
            "selected_cow": current_selected_cow_id,
            "cow_profile": get_cow_profile(current_selected_cow_id),
        })


# --- REAL-TIME VOICE SPEAKER ENDPOINT ---
@app.route("/api/trigger_speaker", methods=["POST"])
def trigger_speaker():
    data = request.get_json() or {}
    status = data.get("status", current_risk_status)
    risk_pct = data.get("risk", current_risk_percentage)
    precaution = data.get("precaution", current_precaution)

    if "Gathering" in str(risk_pct) or "Switching" in str(risk_pct):
        speech_text = "The system is gathering historical data. Please wait."
    else:
        speech_text = f"Attention farmer. The mastitis risk status is {status} at {risk_pct}. Recommended action: {precaution}"

    # Fire voice generation and PCM streaming in background thread
    threading.Thread(
        target=speak_to_esp32, args=(speech_text,), daemon=True
    ).start()

    return jsonify({"status": "speech_streaming_started", "text": speech_text})


if __name__ == "__main__":
    print(
        "[SERVER] Starting Wi-Fi Farm Logbook on http://localhost:5000 ...\n"
    )
    app.run(host="0.0.0.0", port=5000, debug=False)