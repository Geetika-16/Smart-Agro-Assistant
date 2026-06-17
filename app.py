from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, jsonify)
from mainnew import predict_disease
from mainnew import get_crop_info
from mainnew import get_mistral_solution
import mysql.connector
import base64
import tempfile
from datetime import datetime
import json, os
import pandas as pd
from googletrans import Translator
from PIL import Image
import numpy as np, io
import re
from deep_translator import GoogleTranslator
# ── Optional: uncomment when ready ──────────────────────
# import tensorflow as tf  # or torch for your CNN model

LANGUAGE_CSV = "language_dataset.csv"

def load_languages():
    df = pd.read_csv(LANGUAGE_CSV, encoding="utf-8")
    languages = {}
    for _, row in df.iterrows():
        code = str(row["Translate_code"]).strip()
        if code not in languages:
            languages[code] = {
                "name":        str(row["Language"]).strip(),
                "code":        code,
                "speech_code": str(row["Speech_Code"]).strip(),  # e.g. "ta-IN"
                "tts_code":    str(row["TTS_Code"]).strip(),     # e.g. "ta-IN"
            }
        else:
            current = str(row["Language"]).strip()
            if not current.isascii():
                languages[code]["native"] = current
 
    final_languages = []
    for code, info in languages.items():
        final_languages.append({
            "name":        info["name"],
            "native":      info.get("native", info["name"]),
            "code":        code,
            "speech_code": info.get("speech_code", code),
            "tts_code":    info.get("tts_code",    code),
        })
    return final_languages

app = Flask(__name__)
app.secret_key = 'agro_secret_key_change_this'   # ← Change in production!

def translate_text(text, lang):

    try:
        return GoogleTranslator(
            source="en",
            target=lang
        ).translate(text)

    except:
        return text

@app.route("/set-language", methods=["POST"])
def set_language():

    data = request.get_json()

    session["lang"] = data["language"]

    return {"success": True}

# ── MySQL connection helper ──────────────────────────────
def get_db():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Geeti#2216",
        database="smart_agro_db"
    )
    return conn

@app.route('/testdb')
def testdb():
    conn = get_db()
    conn.close()
    return "Database Connected Successfully"

# ── Translator helper ────────────────────────────────────

#  ROUTES

@app.route('/')
def index():
    languages = load_languages()
    return render_template(
        'index.html',
        languages=languages
    )
@app.route("/testlang")
def testlang():
    return session.get("lang", "NOT FOUND")

@app.route("/api/languages")
def api_languages():
    df = pd.read_csv(LANGUAGE_CSV, encoding="utf-8")
    languages = {}
    for _, row in df.iterrows():
        translate_code = str(row["Translate_code"]).strip()
        if translate_code not in languages:
            languages[translate_code] = {
                "code":        str(row["Speech_Code"]).strip(),  # BCP-47 e.g. "ta-IN"
                "tts_code":    str(row["TTS_Code"]).strip(),     # BCP-47 e.g. "ta-IN"
                "native":      str(row["Language"]).strip(),
            }
    return jsonify(languages)
 

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name  = request.form.get('name', '').strip()
        phone = request.form.get('phone', '').strip()
        dob   = request.form.get('dob', '')

        if not (name and phone and dob):
            flash('All fields are required.', 'error')
            return redirect(url_for('register'))
        
        conn = get_db()
        cur = conn.cursor()
        
        cur.execute(
            """
            INSERT INTO users
            (name, phone, dob)
            VALUES (%s,%s,%s)
            """,
            (name, phone, dob)
            )
        conn.commit()
        conn.close()

        flash(f'Welcome {name}! Account created. Please login.', 'success')
        return redirect(url_for('login'))

    return render_template(
    'register.html',
    selected_lang=session.get("lang","en")
)


@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        phone = request.form.get('phone', '').strip()

        if not phone:
            flash('Please enter your phone number.', 'error')
            return redirect(url_for('login'))

        conn = get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT * FROM users WHERE phone=%s",
            (phone,)
            )
        user = cur.fetchone()
        conn.close()
        
        if not user:
            flash("Phone number not registered", "error")
            return redirect(url_for("login"))

        session['user_id']   = user['id']
        session['user_name'] = user['name']
        session['user_phone']= user['phone']
        session['user_joined'] = str(user.get('joined', ''))
        return redirect(url_for('dashboard'))

    return render_template(
    'login.html',
    selected_lang=session.get("lang","en")
)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/dashboard')
def dashboard():

    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db()
    cur = conn.cursor(dictionary=True)

    # Disease count
    cur.execute("""
        SELECT COUNT(*) AS total
        FROM reports
        WHERE user_id=%s AND type='disease'
    """, (session['user_id'],))
    diseases = cur.fetchone()['total']

    # Crop count
    cur.execute("""
        SELECT COUNT(*) AS total
        FROM reports
        WHERE user_id=%s AND type='crop'
    """, (session['user_id'],))
    crops = cur.fetchone()['total']

    # Total reports
    cur.execute("""
        SELECT COUNT(*) AS total
        FROM reports
        WHERE user_id=%s
    """, (session['user_id'],))
    reports = cur.fetchone()['total']

    conn.close()

    stats = {
        "diseases": diseases,
        "crops": crops,
        "reports": reports
    }
    
    return render_template(
        'dashboard.html',
        stats=stats,
        selected_lang=session.get("lang","en")
    )

@app.route('/disease')
def disease():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template(
    'disease.html',
    selected_lang=session.get("lang","en")
)


@app.route('/crop')
def crop():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template(
    'crop.html',
    selected_lang=session.get("lang","en")
)


@app.route('/reports')
def reports():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # ── Fetch from MySQL ───────────────────────────────
    conn = get_db(); 
    cur = conn.cursor(dictionary=True)
    cur.execute(""" 
                SELECT * 
                FROM reports 
                WHERE user_id=%s 
                ORDER BY created_at DESC
    """, 
    (session['user_id'],))
    reports_list = cur.fetchall(); 
    conn.close()
    # Convert reports into frontend format
    formatted_reports = []

    for row in reports_list:

        formatted_reports.append({

            "id": row["id"],

            "type": row["type"],

            "title": row["title"],

            "summary": row["result"][:100],

            "result": row["result"],

            "date": row["created_at"].strftime("%Y-%m-%d %H:%M"),

            "icon": "🌾" if row["type"] == "crop" else "🌿"

        })

    user = {
        'name': session.get('user_name', 'Farmer'),
        'phone': session.get('user_phone', ''),
        'joined': session.get('user_joined', '2025')
    }

    return render_template (
        'reports.html',
        user=user,
        reports=formatted_reports,
        selected_lang=session.get("lang","en")
        )



# ═══════════════════════════════════════════════════════════
#  API ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.route('/api/detect', methods=['POST'])
def api_detect():

    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    temp_path = None

    try:
        data = request.json

        image_data = data.get("image", "")

        image_bytes = base64.b64decode(image_data.split(",")[1])
 
        # Write to temp file — handle closes automatically at end of 'with'
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_bytes)
            temp_path = tmp.name          # save the path for later use
 
        result       = predict_disease(temp_path)
        disease_name = result["disease_name"]
        treatment    = get_mistral_solution(disease_name)
 
        print("Treatment:", treatment)
 

        english_summary = f"""
Disease Detected: {disease_name}

Confidence: {round(result['confidence'] * 100,2)}%

Organic Treatments:
{chr(10).join(treatment['organic'])}

Chemical Treatments:
{chr(10).join(treatment['chemical'])}
"""
        selected_lang = session.get(
            "lang",
            "en"
        )
        translated_summary = translate_text(
            english_summary,
            selected_lang
        )
        print("Disease API Called")
        print("Image received")
        print("Image length:", len(image_data))

        return jsonify({
            "disease_name": disease_name,
            "confidence": result["confidence"],
            "organic": treatment["organic"],
            "chemical": treatment["chemical"],
            "english_summary": english_summary,
            "translated_result": translated_summary,
            "is_healthy": "healthy" in disease_name.lower()
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500
    
    finally:
        # Always delete the temp file, whether the request succeeded or failed
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
    
    
@app.route('/api/crop', methods=['POST'])
def api_crop():

    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    try:

        data = request.json

        crop_name = data.get(
            "crop_name",
            ""
        )
        crop_name = re.sub(
            r'[^a-zA-Z0-9\s]',
            '',
            crop_name
            ).strip()
        print("Crop API Called")
        print(data)
        print(crop_name)

        crop = get_crop_info(crop_name)
        
        print("Crop result:", crop)

        if crop is None:
            return jsonify({
                "error": "Crop not found"
            })

        english_summary = f"""
Crop Name: {crop['Crop_Name']}
Category: {crop['Category']}
Water Requirement: {crop['Water']}
Sunlight: {crop['Sunlight']}
Soil Type: {crop['Soil_Type']}
Climate: {crop['Best_Climate']}
Harvest Time: {crop['Harvest_Time']}
"""
        selected_lang = session.get(
            "lang",
            "en"
        )
        
        translated_summary = translate_text(
            english_summary,
            selected_lang
        )
        print("ENGLISH =", english_summary)
        print("TRANSLATED =", translated_summary)

        return jsonify({

            "name": crop["Crop_Name"],

            "category": crop["Category"],

            "water": crop["Water"],

            "sunlight": crop["Sunlight"],

            "soil": crop["Soil_Type"],

            "climate": crop["Best_Climate"],

            "harvest_time": crop["Harvest_Time"],

            "english_summary": english_summary,

            "translated_result": translated_summary,

            "tags": [
                crop["Category"]
            ],

            "steps": [
                {
                    "title": "Prepare Soil",
                    "desc": f"Use {crop['Soil_Type']} soil"
                },
                {
                    "title": "Plant",
                    "desc": "Sow seeds properly"
                },
                {
                    "title": "Watering",
                    "desc": f"Provide {crop['Water']} water"
                },
                {
                    "title": "Harvest",
                    "desc": crop["Harvest_Time"]
                }
            ]
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/api/save-report', methods=['POST'])
def api_save_report():

    try:

        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401

        data = request.json

        print("SAVE REPORT CALLED")
        print(data)
        print("USER ID:", session['user_id'])

        conn = get_db()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO reports
            (
            user_id,
            type,
            title,
            result,
            created_at
            )
            VALUES
            (%s,%s,%s,%s,NOW())
        """,
        (
            session['user_id'],
            data.get('type'),
            data.get('crop_name') or data.get('disease_name'),
            data.get('result')
        ))

        conn.commit()

        print("Inserted ID:", cur.lastrowid)

        conn.close()

        return jsonify({'status':'saved'})

    except Exception as e:

        print("SAVE ERROR:", e)

        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/delete-report/<int:report_id>', methods=['DELETE'])
def api_delete_report(report_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    # conn = get_db(); cur = conn.cursor()
    # cur.execute("DELETE FROM reports WHERE id=%s AND user_id=%s", (report_id, session['user_id']))
    # conn.commit(); conn.close()

    return jsonify({'status': 'deleted'})


# ═══════════════════════════════════════════════════════════
#  MySQL Schema — run this once in MySQL Workbench
# ═══════════════════════════════════════════════════════════
MYSQL_SCHEMA = """
CREATE DATABASE IF NOT EXISTS smart_agro_db;
USE smart_agro_db;

CREATE TABLE IF NOT EXISTS users (
    id       INT AUTO_INCREMENT PRIMARY KEY,
    name     VARCHAR(100) NOT NULL,
    phone    VARCHAR(15)  NOT NULL UNIQUE,
    dob      DATE,
    joined   DATE DEFAULT (CURRENT_DATE),
    lang     VARCHAR(5)  DEFAULT 'en',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reports (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    user_id    INT NOT NULL,
    type       ENUM('disease','crop') NOT NULL,
    title      VARCHAR(200),
    result     TEXT,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""

if __name__ == "__main__":
    print("═" * 55)
    print(" Smart Agro Assistant")
    print(" http://127.0.0.1:5000")
    print("═" * 55)

    app.run(debug=False, use_reloader=False)