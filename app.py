import json
import threading
import time
import joblib
import pandas as pd
import serial
from flask import Flask, render_template_string
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Configuration
SERIAL_PORT = "COM3"
BAUD_RATE = 115200
MODEL_FILE = "mastitis_model.pkl"
DATASET_CSV = "mastitis_dataset_with_cow_info.csv"

sensor_buffer = []
buffer_lock = threading.Lock()
active_cow_id = "C0001"

model = None
feature_names = None

def load_cow_dataset_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        metadata_df = df[['Cow_ID', 'Age_Years', 'Breed']].drop_duplicates(subset=['Cow_ID'])
        dataset = {}
        for _, row in metadata_df.iterrows():
            dataset[str(row['Cow_ID'])] = {
                "Age_Years": int(row['Age_Years']),
                "Breed": str(row['Breed'])
            }
        print(f"[SUCCESS] Loaded metadata for {len(dataset)} cows from {csv_path}")
        return dataset
    except Exception as e:
        print(f"[WARNING] Could not load CSV dataset ({e}). Using default dataset.")
        return {
            "C0001": {"Age_Years": 8, "Breed": "Holstein Friesian"},
            "C0002": {"Age_Years": 2, "Breed": "Holstein Friesian"},
        }

COW_DATASET = load_cow_dataset_from_csv(DATASET_CSV)

try:
    artifacts = joblib.load(MODEL_FILE)
    if isinstance(artifacts, dict):
        model = artifacts["model"]
        feature_names = artifacts["feature_names"]
    else:
        model = artifacts
        feature_names = getattr(model, "feature_names_in_", None)
    print(f"[SUCCESS] Loaded ML model from {MODEL_FILE}")
except Exception as e:
    print(f"[WARNING] Could not load model: {e}. Fallback logic enabled.")

@socketio.on('connect')
def handle_connect():
    socketio.emit('init_dataset', COW_DATASET)

@socketio.on('change_cow')
def handle_change_cow(data):
    global active_cow_id
    new_id = data.get('cow_id')
    if new_id in COW_DATASET:
        active_cow_id = new_id

def process_and_predict(data, raw_json_str):
    global sensor_buffer, model, feature_names, active_cow_id

    sensor_cols = ["Body_Temperature_C", "Env_Temperature_C", "Env_Humidity_Percent", "Cows_Movement_Activity",
                   "Milk_Conductivity_mS_cm"]

    with buffer_lock:
        sensor_buffer.append(data)
        if len(sensor_buffer) > 14:
            sensor_buffer.pop(0)
        buffer_snapshot = list(sensor_buffer)

    meta = COW_DATASET.get(active_cow_id, {"Age_Years": 5, "Breed": "Holstein Friesian"})

    if model is not None and feature_names is not None:
        df = pd.DataFrame(buffer_snapshot)
        for col in sensor_cols:
            if col in df.columns:
                df[f"{col}_roll3_mean"] = df[col].rolling(3, min_periods=1).mean()
                df[f"{col}_roll7_mean"] = df[col].rolling(7, min_periods=1).mean()
                df[f"{col}_roll7_std"] = df[col].rolling(7, min_periods=1).std().fillna(0)

        df["Age_Years"] = meta["Age_Years"]
        breed = meta["Breed"]

        for col in feature_names:
            if col.startswith("Breed_"):
                df[col] = 1 if col == f"Breed_{breed}" else 0
            elif col not in df.columns:
                df[col] = 0

        X_latest = df.iloc[[-1]][feature_names]
        try:
            risk_prob = model.predict_proba(X_latest)[0][1]
            risk_pct = round(float(risk_prob) * 100, 2)
        except Exception:
            risk_pct = 12.5
    else:
        ec = data.get("Milk_Conductivity_mS_cm", 4.5)
        temp = data.get("Body_Temperature_C", 38.5)
        age_factor = meta["Age_Years"] * 0.5
        risk_pct = round(min(100.0, max(0.0, (ec - 4.5) * 30 + (temp - 38.5) * 20 + age_factor)), 2)

    if risk_pct > 60:
        status, theme = "HIGH RISK - Inspect Immediately", "high"
        precaution = "Isolate subject. Conduct California Mastitis Test (CMT) and consult vet for targeted antibiotics."
    elif risk_pct > 35:
        status, theme = "MODERATE RISK - Monitor", "medium"
        precaution = "Manually check udder for heat or swelling. Strip milk to check for clots, flakes, or watery texture."
    else:
        status, theme = "LOW RISK - Normal", "low"
        precaution = "Maintain standard post-milking teat dipping and routine environmental hygiene."

    return {
        "cow_id": active_cow_id,
        "age": meta["Age_Years"],
        "breed": meta["Breed"],
        "body_temp": data.get("Body_Temperature_C", 0),
        "env_temp": data.get("Env_Temperature_C", 0),
        "humidity": data.get("Env_Humidity_Percent", 0),
        "activity": data.get("Live_Steps", data.get("Cows_Movement_Activity", 0)),
        "ec": data.get("Milk_Conductivity_mS_cm", 0),
        "accel_x": data.get("Accel_X", 0.0),
        "accel_y": data.get("Accel_Y", 0.0),
        "accel_z": data.get("Accel_Z", 0.0),
        "risk": risk_pct,
        "status": status,
        "theme": theme,
        "precaution": precaution,
        "raw_json": raw_json_str
    }

def serial_reader_thread():
    print(f"[SERIAL] Attempting to connect to hardware on port {SERIAL_PORT} at {BAUD_RATE} baud...")
    while True:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
            print(f"[SUCCESS] Connected to hardware on {SERIAL_PORT}! Listening for data streams...")
            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    print(f"[RECV] {line}")
                if line.startswith("JSON_DATA:"):
                    json_str = line.replace("JSON_DATA:", "")
                    data = json.loads(json_str)
                    payload = process_and_predict(data, json_str)
                    socketio.emit("new_sensor_data", payload)
        except Exception as e:
            print(f"[DISCONNECTED] Port {SERIAL_PORT} unavailable or error ({e}). Retrying in 3 seconds...")
            time.sleep(3)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bovine Telemetry Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.1/socket.io.min.js"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
        .glass-panel { background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2); }
        .theme-low { background: linear-gradient(135deg, #10b981, #059669); color: white; }
        .theme-medium { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; }
        .theme-high { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }
        .terminal-scroll::-webkit-scrollbar-thumb { background: #475569; }
    </style>
</head>
<body class="bg-slate-50 text-slate-800 antialiased p-4 md:p-8 min-h-screen">

    <div class="max-w-7xl mx-auto space-y-6">

        <!-- HEADER -->
        <div class="glass-panel rounded-3xl shadow-sm p-6 flex flex-col md:flex-row justify-between items-center gap-4">
            <div class="flex items-center gap-4">
                <div class="w-12 h-12 bg-blue-100 text-blue-600 rounded-2xl flex items-center justify-center text-2xl shadow-inner">
                    <i class="fa-solid fa-cow"></i>
                </div>
                <div>
                    <h1 class="text-2xl font-bold text-slate-900 tracking-tight">Mastitis Intelligence</h1>
                    <p class="text-slate-500 text-sm font-medium">Real-time Telemetry & ML Inference</p>
                </div>
            </div>

            <div class="flex items-center gap-3 bg-white p-2 rounded-xl shadow-sm border border-slate-200 w-full md:w-auto">
                <div class="pl-3 text-slate-400"><i class="fa-solid fa-magnifying-glass"></i></div>
                <select id="cowSelect" class="bg-transparent border-none text-slate-700 text-sm font-semibold focus:ring-0 w-full md:w-64 outline-none cursor-pointer">
                    <option>Loading dataset...</option>
                </select>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

            <!-- LEFT COLUMN: RISK & METADATA -->
            <div class="lg:col-span-4 space-y-6">

                <!-- PREDICTION BOX -->
                <div id="riskCard" class="rounded-3xl shadow-lg p-8 text-center transition-all duration-500 theme-low relative overflow-hidden flex flex-col items-center">
                    <div class="absolute top-0 right-0 p-4 opacity-20 text-6xl"><i class="fa-solid fa-shield-heart"></i></div>
                    <h2 class="text-xs font-bold uppercase tracking-widest opacity-80 mb-2 relative z-10">7-Day Risk Probability</h2>
                    <div class="text-7xl font-black my-2 tracking-tighter drop-shadow-md relative z-10" id="riskVal">--%</div>
                    <div class="inline-flex items-center gap-2 bg-white/20 backdrop-blur-md px-4 py-2 rounded-full text-sm font-semibold relative z-10 mb-6" id="statusText">
                        <i class="fa-solid fa-circle-notch fa-spin"></i> Awaiting Data
                    </div>

                    <!-- DYNAMIC PRECAUTION PANEL -->
                    <div class="w-full text-sm font-medium bg-black/10 backdrop-blur-md rounded-2xl p-4 text-left border border-white/20 relative z-10">
                        <div class="font-bold mb-1 opacity-90 tracking-wide flex items-center gap-2">
                            <i class="fa-solid fa-user-doctor"></i> Action Plan
                        </div>
                        <div id="precautionText" class="opacity-90 leading-relaxed text-xs sm:text-sm">
                            Waiting for telemetry stream...
                        </div>
                    </div>
                </div>

                <!-- SUBJECT INFO -->
                <div class="glass-panel rounded-3xl shadow-sm p-6">
                    <h2 class="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4"><i class="fa-solid fa-clipboard-list mr-2"></i>Subject Profile</h2>
                    <div class="space-y-4">
                        <div class="flex justify-between items-center">
                            <span class="text-slate-500 flex items-center gap-2"><i class="fa-solid fa-fingerprint text-slate-300 w-4"></i> ID</span>
                            <span class="font-bold text-slate-900 bg-slate-100 px-3 py-1 rounded-lg" id="metaId">--</span>
                        </div>
                        <div class="flex justify-between items-center">
                            <span class="text-slate-500 flex items-center gap-2"><i class="fa-solid fa-calendar-days text-slate-300 w-4"></i> Age</span>
                            <span class="font-bold text-slate-900" id="metaAge">-- yrs</span>
                        </div>
                        <div class="flex justify-between items-center">
                            <span class="text-slate-500 flex items-center gap-2"><i class="fa-solid fa-dna text-slate-300 w-4"></i> Breed</span>
                            <span class="font-bold text-slate-900 text-right w-40 truncate" id="metaBreed">--</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- RIGHT COLUMN: SENSORS -->
            <div class="lg:col-span-8 grid grid-cols-1 md:grid-cols-2 gap-6">

                <div class="glass-panel rounded-3xl shadow-sm p-6 flex items-center justify-between group hover:shadow-md transition-shadow">
                    <div>
                        <span class="text-slate-400 text-xs font-bold uppercase tracking-widest mb-1 block">Body Temp</span>
                        <span class="text-4xl font-black text-slate-800 tracking-tight" id="btVal">--</span>
                        <span class="text-slate-500 font-semibold ml-1">&deg;C</span>
                    </div>
                    <div class="w-14 h-14 bg-red-50 text-red-500 rounded-full flex items-center justify-center text-2xl group-hover:scale-110 transition-transform">
                        <i class="fa-solid fa-temperature-half"></i>
                    </div>
                </div>

                <div class="glass-panel rounded-3xl shadow-sm p-6 flex items-center justify-between group hover:shadow-md transition-shadow border-l-4 border-blue-500">
                    <div>
                        <span class="text-slate-400 text-xs font-bold uppercase tracking-widest mb-1 block">Milk Conductivity</span>
                        <span class="text-4xl font-black text-blue-600 tracking-tight" id="ecVal">--</span>
                        <span class="text-slate-500 font-semibold ml-1">mS/cm</span>
                    </div>
                    <div class="w-14 h-14 bg-blue-50 text-blue-500 rounded-full flex items-center justify-center text-2xl group-hover:scale-110 transition-transform">
                        <i class="fa-solid fa-flask"></i>
                    </div>
                </div>

                <div class="glass-panel rounded-3xl shadow-sm p-6 flex items-center justify-between group hover:shadow-md transition-shadow">
                    <div>
                        <span class="text-slate-400 text-xs font-bold uppercase tracking-widest mb-1 block">Activity (Steps)</span>
                        <span class="text-4xl font-black text-slate-800 tracking-tight" id="actVal">--</span>
                        <span class="text-slate-500 font-semibold ml-1">steps</span>
                    </div>
                    <div class="w-14 h-14 bg-purple-50 text-purple-500 rounded-full flex items-center justify-center text-2xl group-hover:scale-110 transition-transform">
                        <i class="fa-solid fa-wave-square"></i>
                    </div>
                </div>

                <div class="glass-panel rounded-3xl shadow-sm p-6 flex items-center justify-between group hover:shadow-md transition-shadow">
                    <div>
                        <span class="text-slate-400 text-xs font-bold uppercase tracking-widest mb-1 block">Ambient (T / RH)</span>
                        <span class="text-3xl font-black text-slate-800 tracking-tight" id="ambVal">-- / --</span>
                    </div>
                    <div class="w-14 h-14 bg-emerald-50 text-emerald-500 rounded-full flex items-center justify-center text-2xl group-hover:scale-110 transition-transform">
                        <i class="fa-solid fa-cloud-sun"></i>
                    </div>
                </div>

                <!-- RAW ACCELEROMETER PANEL -->
                <div class="glass-panel rounded-3xl shadow-sm p-6 flex items-center justify-between group hover:shadow-md transition-shadow md:col-span-2 border-l-4 border-indigo-500">
                    <div>
                        <span class="text-slate-400 text-xs font-bold uppercase tracking-widest mb-1 block">Raw Accelerometer (X, Y, Z)</span>
                        <span class="text-xl font-black text-slate-800 tracking-tight font-mono" id="imuVal">X: 0.00 | Y: 0.00 | Z: 0.00</span>
                        <span class="text-slate-500 font-semibold ml-2 text-xs">m/s&sup2;</span>
                    </div>
                    <div class="w-14 h-14 bg-indigo-50 text-indigo-500 rounded-full flex items-center justify-center text-2xl group-hover:scale-110 transition-transform">
                        <i class="fa-solid fa-arrows-up-down-left-right"></i>
                    </div>
                </div>

                <!-- TERMINAL -->
                <div class="md:col-span-2 bg-slate-900 rounded-3xl shadow-xl overflow-hidden border border-slate-800 mt-2">
                    <div class="bg-slate-800/80 px-5 py-3 border-b border-slate-700/50 flex items-center justify-between">
                        <div class="flex items-center gap-2">
                            <div class="w-3 h-3 rounded-full bg-red-500"></div>
                            <div class="w-3 h-3 rounded-full bg-yellow-500"></div>
                            <div class="w-3 h-3 rounded-full bg-green-500"></div>
                            <span class="text-slate-400 text-xs ml-3 font-mono font-medium tracking-wider">ttyUSB0 - Raw Stream</span>
                        </div>
                        <div class="text-slate-500 text-xs font-mono" id="lastUpdate">Listening...</div>
                    </div>
                    <div id="terminal" class="p-5 h-48 overflow-y-auto font-mono text-xs sm:text-sm text-emerald-400 leading-relaxed space-y-1.5 terminal-scroll">
                        <div class="text-slate-500">Initialize socket connection... OK</div>
                    </div>
                </div>

            </div>
        </div>
    </div>

    <script>
        const socket = io();
        const cowSelect = document.getElementById('cowSelect');
        const terminal = document.getElementById('terminal');

        socket.on('init_dataset', function(dataset) {
            cowSelect.innerHTML = '';
            for (const [cow_id, meta] of Object.entries(dataset)) {
                const option = document.createElement('option');
                option.value = cow_id;
                option.text = `${cow_id} | ${meta.Breed} (${meta.Age_Years}y)`;
                cowSelect.appendChild(option);
            }
        });

        cowSelect.addEventListener('change', function() {
            const selectedId = this.value;
            socket.emit('change_cow', { cow_id: selectedId });
            logToTerminal(`> CMD: Context switched to subject ${selectedId}`, "text-blue-400");
        });

        function logToTerminal(msg, colorClass = "text-emerald-400") {
            const timestamp = new Date().toLocaleTimeString([], {hour12: false, hour: '2-digit', minute:'2-digit', second:'2-digit'});
            const logEntry = document.createElement('div');
            logEntry.innerHTML = `<span class="text-slate-500">[${timestamp}]</span> <span class="${colorClass}">${msg}</span>`;
            terminal.appendChild(logEntry);
            if (terminal.childElementCount > 40) terminal.removeChild(terminal.firstChild);
            terminal.scrollTop = terminal.scrollHeight;
            document.getElementById('lastUpdate').innerText = "Last packet: " + timestamp;
        }

        socket.on('new_sensor_data', function(data) {
            document.getElementById('metaId').innerText = data.cow_id;
            document.getElementById('metaAge').innerText = data.age;
            document.getElementById('metaBreed').innerText = data.breed;

            document.getElementById('riskVal').innerText = data.risk + "%";

            const riskCard = document.getElementById('riskCard');
            const statusText = document.getElementById('statusText');

            riskCard.className = `rounded-3xl shadow-lg p-8 text-center transition-all duration-500 relative overflow-hidden flex flex-col items-center theme-${data.theme}`;
            statusText.innerHTML = `<i class="fa-solid fa-circle text-xs mr-1 drop-shadow-sm"></i> ${data.status}`;
            document.getElementById('precautionText').innerText = data.precaution;

            document.getElementById('btVal').innerText = data.body_temp;
            document.getElementById('ecVal').innerText = data.ec;
            document.getElementById('actVal').innerText = data.activity;
            document.getElementById('ambVal').innerHTML = `${data.env_temp}&deg;C / ${data.humidity}%`;

            // Update IMU Accelerometer Panel
            const x = (data.accel_x || 0).toFixed(2);
            const y = (data.accel_y || 0).toFixed(2);
            const z = (data.accel_z || 0).toFixed(2);
            document.getElementById('imuVal').innerText = `X: ${x} | Y: ${y} | Z: ${z}`;

            logToTerminal(data.raw_json);

            if(cowSelect.value !== data.cow_id) { cowSelect.value = data.cow_id; }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

if __name__ == "__main__":
    t = threading.Thread(target=serial_reader_thread, daemon=True)
    t.start()
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)