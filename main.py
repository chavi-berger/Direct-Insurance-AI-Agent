import os
import asyncio
import json
import shutil
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. טעינת משתני סביבה
load_dotenv()

app = FastAPI(
    title="Direct Insurance AI Recruitment Agent - Orchestrator",
    description="מערכת מאוחדת לניתוח קולי ונתוני רשת עבור ביטוח ישיר"
)

# 2. הגדרת CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === קישור לפרויקט הראשון (Web Search) ===
PROJECT_1_URL = "https://vv8xmj-8080.csb.app/"
# =========================================

# 3. אתחול הלקוח של גוגל
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env file!")
client = genai.Client(api_key=api_key)


async def run_ai_analysis(file_path: str, candidate_full_name: str):
    """פונקציה לניתוח קובץ האודיו מול Gemini"""

    print(f"📤 Uploading {file_path} to Gemini...")
    uploaded_file = client.files.upload(file=file_path)

    while uploaded_file.state.name == "PROCESSING":
        print("⏳ AI is processing the audio...")
        await asyncio.sleep(3)
        uploaded_file = client.files.get(name=uploaded_file.name)

    if uploaded_file.state.name == "FAILED":
        raise Exception("Audio processing failed on Google servers.")

    prompt = f"""
    Role: Senior HR Analyst at 'Direct Insurance'.
    Task: Analyze the audio recording of the candidate: {candidate_full_name}.

    Evaluation Instructions:
    - Listen for vocal cues: confidence, empathy, service-orientation, and energy.
    - Provide scores (1-10) for each criteria in the JSON.
    - Write a deep, professional summary in HEBREW.
    - Return ONLY a JSON object.
    - Calculate the match_percentage as an average of all detailed_scores.

    STRICT JSON STRUCTURE:
    {{
        "status": "success",
        "candidate": "{candidate_full_name}",
        "analysis_result": {{
            "candidate_name": "{candidate_full_name}",
            "match_percentage": 0,
            "detailed_scores": {{
                "Fluent_Communication": 0,
                "Punctuality": 0,
                "Integrity_Reliability": 0,
                "Career_Stability": 0,
                "Efficiency_Agility": 0,
                "High_Energy_Motivation": 0,
                "Adaptability_Inclusion": 0,
                "Target_Age_Group": 0,
                "Clean_Record": 0,
                "Team_Player": 0,
                "Active_Listening": 0,
                "Customer_Centricity": 0,
                "Overall": 0
            }},
            "summary": "Full professional Hebrew evaluation...",
            "sources": [],
            "recommendation": "Proceed / Hold / Reject"
        }}
    }}
    """

    print("🧠 Analyzing Audio with Gemini...")
    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=[uploaded_file, prompt],
        config=types.GenerateContentConfig(
            response_mime_type='application/json'
        )
    )

    return json.loads(response.text)


@app.post("/analyze")
async def analyze_candidate(
        first_name: str = Form(...),
        last_name: str = Form(...),
        email: str = Form(None),
        audio: UploadFile = File(...)
):
    """
    Endpoint מרכזי: מקבל אודיו מה-React ומביא JSON מפרויקט 1
    """
    full_name = f"{first_name} {last_name}"
    temp_file_name = f"temp_{audio.filename}"
    
    # --- שלב 1: קבלת JSON מפרויקט 1 (Web Search) ---
    print(f"🌐 Calling Project 1 (Web Search) for: {full_name}")
    project_1_data = {}
    try:
        web_res = requests.post(PROJECT_1_URL, json={
            "first_name": first_name,
            "last_name": last_name,
            "email": email
        }, timeout=60)
        
        if web_res.status_code == 200:
            project_1_data = web_res.json()
        else:
            project_1_data = {"status": "error", "message": "Project 1 search failed"}
    except Exception as e:
        project_1_data = {"status": "error", "message": f"Connection to Project 1 failed: {str(e)}"}

    # --- שלב 2: שמירת קובץ וניתוח אודיו (Project 2) ---
    with open(temp_file_name, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    try:
        # הרצת ניתוח האודיו
        project_2_data = await run_ai_analysis(temp_file_name, full_name)

        # --- שלב 3: החזרת שני ה-JSON-ים המקוריים ל-React ---
        return {
            "web_search_project": project_1_data,   # ה-JSON המקורי מפרויקט 1
            "audio_analysis_project": project_2_data # ה-JSON המקורי מפרויקט 2
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)


@app.get("/")
def read_root():
    return {"message": "Unified Agent is Online!"}


if __name__ == "__main__":
    import uvicorn
    # הרצה על 0.0.0.0 ופורט 8080 עבור CodeSandbox
    uvicorn.run(app, host="0.0.0.0", port=8080)