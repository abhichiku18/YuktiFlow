import os
import io
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, static_folder="static", template_folder="templates")

# -------------------------
# Load models (must exist in project directory)
# -------------------------
prophet_model = joblib.load('prophet_model.pkl')
rf_model = joblib.load('rf_model.pkl')
log_model = joblib.load('log_model.pkl')
kmeans_model = joblib.load('kmeans_model.pkl')

# -------------------------
# Try loading encoders (optional)
# -------------------------
def load_encoder(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None

le_product_saved = load_encoder('le_product.pkl')
le_area_saved = load_encoder('le_area.pkl')

# -------------------------
# Helper functions
# -------------------------
def ensure_numeric_encoding(df, col, saved_encoder=None):
    """
    Encode categorical column using saved encoder if possible,
    else fit a new one.
    """
    if saved_encoder is not None:
        try:
            enc = saved_encoder
            encoded = enc.transform(df[col].astype(str))
            return encoded, enc, True
        except Exception:
            pass
    enc = LabelEncoder()
    encoded = enc.fit_transform(df[col].astype(str))
    return encoded, enc, False


def map_kmeans_clusters(labels, df_units):
    """
    Map KMeans clusters to 'Fast Moving' or 'Slow Moving'
    based on average units sold.
    """
    df_tmp = pd.DataFrame({'cluster': labels, 'Units_Sold': df_units})
    means = df_tmp.groupby('cluster')['Units_Sold'].mean().to_dict()
    sorted_clusters = sorted(means.items(), key=lambda x: x[1], reverse=True)
    mapping = {}
    if len(sorted_clusters) >= 1:
        mapping[sorted_clusters[0][0]] = 'Fast Moving'
        for cid, _ in sorted_clusters[1:]:
            mapping[cid] = 'Slow Moving'
    else:
        for cid in set(labels):
            mapping[cid] = 'Unknown'
    mapped = [mapping.get(int(c), 'Unknown') for c in labels]
    return mapped, mapping

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500


# Routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {e}"}), 400

    df_display = df.copy()
    response_meta = {"encoder_used_warning": []}
    results = {}

    # Prophet forecast 
    try:
        if ('Date' in df.columns) and ('Units_Sold' in df.columns):
            df_prophet = df[['Date', 'Units_Sold']].rename(columns={'Date': 'ds', 'Units_Sold': 'y'})
            df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
            forecast = prophet_model.predict(df_prophet)
            prophet_out = forecast[['ds', 'yhat']].tail(30).copy()
            prophet_out['ds'] = prophet_out['ds'].astype(str)
            results['prophet'] = prophet_out.to_dict(orient='records')
        else:
            results['prophet'] = []
            response_meta["encoder_used_warning"].append(
                "Prophet: CSV must contain 'Date' and 'Units_Sold' columns."
            )
    except Exception as e:
        results['prophet'] = []
        response_meta["encoder_used_warning"].append(f"Prophet error: {e}")

    # Random Forest 
    rf_preds = []
    try:
        if set(['Product', 'Area', 'Stockout_Risk']).issubset(df.columns):
            X_rf = df[['Product', 'Area', 'Stockout_Risk']].copy()
            encoded_prod, prod_enc, used_saved_prod = ensure_numeric_encoding(X_rf, 'Product', le_product_saved)
            X_rf['Product_enc'] = encoded_prod
            if not used_saved_prod:
                response_meta["encoder_used_warning"].append("RandomForest: Product encoder fitted on new data.")

            encoded_area, area_enc, used_saved_area = ensure_numeric_encoding(X_rf, 'Area', le_area_saved)
            X_rf['Area_enc'] = encoded_area
            if not used_saved_area:
                response_meta["encoder_used_warning"].append("RandomForest: Area encoder fitted on new data.")

            if hasattr(rf_model, "feature_names_in_"):
                X_final = X_rf[list(rf_model.feature_names_in_)]
            else:
                X_final = X_rf[['Product_enc', 'Area_enc', 'Stockout_Risk']]

            rf_vals = rf_model.predict(X_final)
            rf_preds = [round(float(v), 2) for v in rf_vals]
        else:
            response_meta["encoder_used_warning"].append("RandomForest: Missing columns.")
    except Exception as e:
        response_meta["encoder_used_warning"].append(f"RandomForest error: {e}")

    results['rf'] = rf_preds

    #Logistic Regression 
    log_preds = []
    log_labels = []
    try:
        if set(['Product', 'Area', 'Units_Sold']).issubset(df.columns):
            X_log = df[['Product', 'Area', 'Units_Sold']].copy()
            encoded_prod, prod_enc2, used_saved_prod2 = ensure_numeric_encoding(X_log, 'Product', le_product_saved)
            X_log['Product_enc'] = encoded_prod
            encoded_area2, area_enc2, used_saved_area2 = ensure_numeric_encoding(X_log, 'Area', le_area_saved)
            X_log['Area_enc'] = encoded_area2

            if hasattr(log_model, "feature_names_in_"):
                X_final_log = X_log[list(log_model.feature_names_in_)]
            else:
                X_final_log = X_log[['Product_enc', 'Area_enc', 'Units_Sold']]

            lp = log_model.predict(X_final_log)
            log_preds = [int(v) for v in lp]
            log_labels = ['High' if v == 1 else 'Low' for v in log_preds]
        else:
            response_meta["encoder_used_warning"].append("Logistic Regression: Missing columns.")
    except Exception as e:
        response_meta["encoder_used_warning"].append(f"Logistic Regression error: {e}")

    results['log_numeric'] = log_preds
    results['log'] = log_labels

    # KMeans 
    km_preds = []
    km_labels = []
    try:
        if 'Units_Sold' in df.columns:
            X_km = df[['Units_Sold']].copy()
            km_raw = kmeans_model.predict(X_km)
            km_preds = [int(v) for v in km_raw]
            km_labels, mapping = map_kmeans_clusters(km_preds, df['Units_Sold'])
        else:
            response_meta["encoder_used_warning"].append("KMeans: CSV must contain 'Units_Sold'.")
    except Exception as e:
        response_meta["encoder_used_warning"].append(f"KMeans error: {e}")

    results['kmeans_numeric'] = km_preds
    results['kmeans'] = km_labels

    # Build Output Table 
    out_df = df_display.copy()
    out_df['RF_Prediction'] = rf_preds if len(rf_preds) == len(out_df) else [None]*len(out_df)
    out_df['LogReg_Prediction'] = log_labels if len(log_labels) == len(out_df) else [None]*len(out_df)
    out_df['KMeans_Prediction'] = km_labels if len(km_labels) == len(out_df) else [None]*len(out_df)

    results['table'] = out_df.to_dict(orient='records')
    results['meta'] = response_meta

    #Make JSON Safe 
    def make_json_safe(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, list)):
            return [make_json_safe(o) for o in obj]
        elif isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        else:
            return obj

    results = make_json_safe(results)
    return jsonify(results)


@app.route('/download_predictions', methods=['POST'])
def download_predictions():
    data = request.get_json()
    if not data or 'table' not in data:
        return jsonify({"error": "No table provided"}), 400
    df_out = pd.DataFrame(data['table'])
    buf = io.StringIO()
    df_out.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='predictions_with_models.csv'
    )


if __name__ == '__main__':
    app.run(debug=True)
