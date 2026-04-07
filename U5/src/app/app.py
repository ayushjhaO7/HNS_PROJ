import pandas as pd
import numpy as np
import os
import json
import joblib
import hashlib
from flask import Flask, request, jsonify, send_from_directory

# ── Dynamic Service Configuration ──
SERVICE_TYPE = os.getenv("SERVICE_TYPE", "analytics").lower()
PORT = int(os.getenv("PORT", 5000))

# ── Path Configuration ──
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
SHARED_DIR = os.getenv("SHARED_DIR", "/app/shared")

app = Flask(__name__, static_folder=STATIC_DIR)

# ── Shared Data Paths ──
MODEL_PATH = os.path.join(SHARED_DIR, "model.pkl")
SCALER_PATH = os.path.join(SHARED_DIR, "scaler.pkl")
META_PATH = os.path.join(SHARED_DIR, "model_meta.joblib")
FEATURE_CSV_PATH = os.path.join(SHARED_DIR, "cleaned_crime_features.csv")
DISTRICT_RISK_PATH = os.path.join(SHARED_DIR, "district_risks.json")

# Primary dataset path (now mounted in /app/data via docker-compose)
RAW_DATA_PATH = "/app/data/Crimes_in_india_2001-2013.csv"
FALLBACK_DATA_PATH = os.path.join(SHARED_DIR, "Crimes_in_india_2001-2013.csv")

model = None
scaler = None
meta = None
feature_df = None
district_risks = None
spark = None

STATE_COORDS = {
    "ANDHRA PRADESH": [15.9129, 79.7400], "ARUNACHAL PRADESH": [28.2180, 94.7278],
    "ASSAM": [26.2006, 92.9376], "BIHAR": [25.0961, 85.3131], "CHHATTISGARH": [21.2787, 81.8661],
    "GOA": [15.2993, 74.1240], "GUJARAT": [22.2587, 71.1924], "HARYANA": [29.0588, 76.0856],
    "HIMACHAL PRADESH": [31.1048, 77.1734], "JAMMU & KASHMIR": [33.7782, 76.5762],
    "JHARKHAND": [23.6102, 85.2799], "KARNATAKA": [15.3173, 75.7139], "KERALA": [10.8505, 76.2711],
    "MADHYA PRADESH": [22.9734, 78.6569], "MAHARASHTRA": [19.7515, 75.7139],
    "MANIPUR": [24.6637, 93.9063], "MEGHALAYA": [25.4670, 91.3662], "MIZORAM": [23.1645, 92.9376],
    "NAGALAND": [26.1584, 94.5624], "ODISHA": [20.9517, 85.0985], "PUNJAB": [31.1471, 75.3412],
    "RAJASTHAN": [27.0238, 74.2179], "SIKKIM": [27.5330, 88.5122], "TAMIL NADU": [11.1271, 78.6569],
    "TELANGANA": [18.1124, 79.0193], "TRIPURA": [23.9408, 91.9882], "UTTAR PRADESH": [26.8467, 80.9462],
    "UTTARAKHAND": [30.0668, 79.0193], "WEST BENGAL": [22.9868, 87.8550], "DELHI UT": [28.6139, 77.2090]
}

STATE_NAME_MAP = {
    "ANDHRA PRADESH": "Andhra Pradesh", "ARUNACHAL PRADESH": "Arunachal Pradesh",
    "ASSAM": "Assam", "BIHAR": "Bihar", "CHHATTISGARH": "Chhattisgarh",
    "GOA": "Goa", "GUJARAT": "Gujarat", "HARYANA": "Haryana",
    "HIMACHAL PRADESH": "Himachal Pradesh", "JAMMU & KASHMIR": "Jammu and Kashmir",
    "JHARKHAND": "Jharkhand", "KARNATAKA": "Karnataka", "KERALA": "Kerala",
    "MADHYA PRADESH": "Madhya Pradesh", "MAHARASHTRA": "Maharashtra",
    "MANIPUR": "Manipur", "MEGHALAYA": "Meghalaya", "MIZORAM": "Mizoram",
    "NAGALAND": "Nagaland", "ODISHA": "Orissa", "PUNJAB": "Punjab",
    "RAJASTHAN": "Rajasthan", "SIKKIM": "Sikkim", "TAMIL NADU": "Tamil Nadu",
    "TRIPURA": "Tripura", "UTTAR PRADESH": "Uttar Pradesh", "UTTARAKHAND": "Uttaranchal",
    "WEST BENGAL": "West Bengal", "DELHI": "Delhi", "DELHI UT": "Delhi",
    "CHANDIGARH": "Chandigarh", "PUDUCHERRY": "Puducherry",
    "A & N ISLANDS": "Andaman and Nicobar", "D & N HAVELI": "Dadra and Nagar Haveli",
    "DAMAN & DIU": "Daman and Diu", "LAKSHADWEEP": "Lakshadweep",
}

RISK_SCORE_MAP = {
    "High Risk": 3,
    "Medium Risk": 2,
    "Low Risk": 1
}

def get_jittered_coords(state, district):
    """Generates consistent district-level coordinates based on state centroid."""
    base = STATE_COORDS.get(state.upper(), [20.5937, 78.9629])
    seed = int(hashlib.md5(district.upper().encode()).hexdigest(), 16) % 1000
    lat_off = (seed / 1000.0 - 0.5) * 2.5
    lng_off = ((seed * 1.3) % 1000 / 1000.0 - 0.5) * 2.5
    return [base[0] + lat_off, base[1] + lng_off]

def load_data():
    global model, scaler, meta, feature_df, district_risks, spark
    try:
        if all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, META_PATH]):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            meta = joblib.load(META_PATH)
        
        if os.path.exists(FEATURE_CSV_PATH):
            feature_df = pd.read_csv(FEATURE_CSV_PATH)

        if SERVICE_TYPE == "prediction":
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.appName("AnalyticalEngine").master("local[*]").getOrCreate()
            
            # Use Mounted Path or Fallback
            active_path = RAW_DATA_PATH if os.path.exists(RAW_DATA_PATH) else FALLBACK_DATA_PATH
            
            if os.path.exists(active_path):
                print(f"[SPARK] Loading dataset from: {active_path}")
                full_df = spark.read.csv(active_path, header=True, inferSchema=True)
                full_df = full_df.filter("UPPER(DISTRICT) NOT LIKE '%TOTAL%'")
                full_df.createOrReplaceTempView("crimes")
                print("[SPARK] View 'crimes' registered successfully.")
            else:
                print(f"[ERROR] Dataset NOT found at {RAW_DATA_PATH} or {FALLBACK_DATA_PATH}. View 'crimes' registration skipped.")

        if os.path.exists(DISTRICT_RISK_PATH):
            with open(DISTRICT_RISK_PATH, 'r') as f:
                district_risks = json.load(f)
        
        print(f"[{SERVICE_TYPE.upper()}] Data Sync Complete. Feature DF: {feature_df is not None}")
        return True
    except Exception as e:
        print(f"[{SERVICE_TYPE.upper()}] Initializing Error: {e}")
        return False

load_data()

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(app.static_folder, f"{SERVICE_TYPE}.html")

@app.route("/api/hierarchy", methods=["GET"])
def get_hierarchy():
    if feature_df is None: return jsonify({})
    hierarchy = feature_df.groupby('STATE/UT')['DISTRICT'].unique().to_dict()
    return jsonify({k: list(v) for k, v in hierarchy.items()})

@app.route("/api/district-intensity", methods=["GET"])
def get_district_intensity():
    global feature_df
    # LAZY LOAD: If data wasn't ready at startup, try one more time
    if feature_df is None:
        load_data()
    
    if feature_df is None: 
        print("[API] Intensity requested but feature_df is still MISSING.")
        return jsonify([])
        
    crime_cols = meta['features'] if meta else [c for c in feature_df.columns if c not in ['STATE/UT', 'DISTRICT']]
    df_copy = feature_df.copy()
    df_copy['total'] = df_copy[crime_cols].sum(axis=1)
    records = []
    for _, row in df_copy.iterrows():
        coords = get_jittered_coords(row['STATE/UT'], row['DISTRICT'])
        records.append({
            "district": row['DISTRICT'], "state": row['STATE/UT'],
            "total": float(row['total']), "lat": coords[0], "lng": coords[1]
        })
    return jsonify(records)

@app.route("/api/query", methods=["POST"])
def run_query():
    if not spark: return jsonify({"error": "Spark Engine not initialized"}), 500
    # Check if view exists
    try:
        spark.sql("SELECT 1 FROM crimes LIMIT 1")
    except:
        return jsonify({"error": "Dataset 'crimes' not found. Ensure file is mounted."}), 404

    q = request.json.get("query")
    if not q.upper().strip().startswith("SELECT"): return jsonify({"error": "Read-only"}), 403
    try:
        res = spark.sql(q).limit(50).toPandas()
        return jsonify(res.to_dict(orient='records'))
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/state-risk", methods=["GET"])
def get_state_risk():
    if district_risks is None: return jsonify({})
    
    state_agg = {}
    
    # 1. Aggregate counts from feature_df (per-feature breakdown for tooltips)
    state_counts = {}
    state_feature_breakdown = {}
    if feature_df is not None:
        crime_cols = meta['features'] if meta else [c for c in feature_df.columns if c not in ['STATE/UT', 'DISTRICT']]
        df_copy = feature_df.copy()
        df_copy['total'] = df_copy[crime_cols].sum(axis=1)
        
        # Total crimes per state
        counts_df = df_copy.groupby('STATE/UT')['total'].sum().to_dict()
        
        # Per-feature breakdown per state
        feature_agg = df_copy.groupby('STATE/UT')[crime_cols].sum()
        
        for s in counts_df:
            geo_name = STATE_NAME_MAP.get(s.upper())
            if geo_name:
                state_counts[geo_name] = state_counts.get(geo_name, 0) + int(counts_df[s])
                if geo_name not in state_feature_breakdown:
                    state_feature_breakdown[geo_name] = {}
                for col in crime_cols:
                    state_feature_breakdown[geo_name][col] = (
                        state_feature_breakdown[geo_name].get(col, 0) + int(feature_agg.loc[s, col])
                    )

    # 2. Aggregate risks from district_risks using AVERAGE (not max)
    state_risk_accum = {}  # {geo_name: {"sum": 0, "count": 0}}
    for key, info in district_risks.items():
        state_name = key.split("|")[0]
        risk_score = RISK_SCORE_MAP.get(info['risk_level'], 0)
        
        geo_name = STATE_NAME_MAP.get(state_name.upper())
        if geo_name:
            if geo_name not in state_risk_accum:
                state_risk_accum[geo_name] = {"sum": 0, "count": 0}
            state_risk_accum[geo_name]["sum"] += risk_score
            state_risk_accum[geo_name]["count"] += 1
    
    # 3. Compute average risk and classify
    for geo_name, accum in state_risk_accum.items():
        avg_risk = accum["sum"] / accum["count"] if accum["count"] > 0 else 0
        
        # Manual override: Force major high-crime states to High Risk (Red)
        force_high = ["Uttar Pradesh", "Maharashtra", "Bihar", "Madhya Pradesh", "Delhi"]
        
        # Thresholds further lowered for maximum sensitivity:
        if geo_name in force_high or avg_risk >= 1.5:
            final_score, final_level = 3, "High Risk"
        elif avg_risk >= 1.2:
            final_score, final_level = 2, "Medium Risk"
        else:
            final_score, final_level = 1, "Low Risk"
        
        state_agg[geo_name] = {
            "risk_score": final_score, 
            "risk_level": final_level, 
            "total_crimes": state_counts.get(geo_name, 0),
            "features": state_feature_breakdown.get(geo_name, {}),
            "avg_risk": round(avg_risk, 2)
        }
    
    # Telangana Special Fix
    if "Telangana" not in state_agg and "Andhra Pradesh" in state_agg:
        state_agg["Telangana"] = state_agg["Andhra Pradesh"].copy()

    return jsonify(state_agg)

@app.route("/api/predict-district", methods=["GET"])
def predict_district():
    state, dist = request.args.get('state'), request.args.get('district')
    key = f"{state}|{dist}"
    if district_risks and key in district_risks and feature_df is not None:
        row = feature_df[(feature_df['STATE/UT']==state) & (feature_df['DISTRICT']==dist)]
        if not row.empty:
            return jsonify({
                "risk": district_risks[key]["risk_level"],
                "features": {c: float(row[c].values[0]) for c in meta['features']}
            })
    return jsonify({"error": "Not found"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
