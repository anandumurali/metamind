# voice_assistant/ai_engine.py

import os
import tempfile
import speech_recognition as sr
import pyttsx3
import cv2
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
import logging
import threading
import time
import json
import random
import subprocess

from vosk import Model, KaldiRecognizer
from deepface import DeepFace

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

tts_engine = None
try:
    tts_engine = pyttsx3.init()
    rate = tts_engine.getProperty('rate')
    logging.info(f"Default TTS speech rate: {rate} words per minute.")
    tts_engine.setProperty('rate', 170)
    logging.info("Successfully initialized pyttsx3 engine.")
except Exception as e:
    logging.error(f"Failed to initialize pyttsx3 engine: {e}")
    raise

main_recognizer = sr.Recognizer()
microphone = sr.Microphone()

VOSK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "vosk-model-small-en-in-0.4")

if not os.path.exists(VOSK_MODEL_PATH):
    logging.error(f"Vosk model not found at: {VOSK_MODEL_PATH}")
    raise FileNotFoundError(f"Vosk model not found at {VOSK_MODEL_PATH}")

vosk_model = None
try:
    vosk_model = Model(VOSK_MODEL_PATH)
    logging.info(f"Successfully loaded Vosk model from: {VOSK_MODEL_PATH}")
except Exception as e:
    logging.error(f"Failed to load Vosk model: {e}")
    raise

WAKE_WORD_KEYPHRASE = "metamind"
assistant_active = threading.Event()
stop_audio_playback_flag = threading.Event()

current_emotion_state = "neutral"  # Store detected emotion globally

gemini_api_key = os.environ.get("GOOGLE_API_KEY")
if not gemini_api_key:
    logging.warning("GOOGLE_API_KEY not set in .env file.")

genai.configure(api_key=gemini_api_key)
GEMINI_MODEL_NAME = "gemini-1.5-flash"
try:
    if gemini_api_key:
        available_models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
        if GEMINI_MODEL_NAME not in available_models:
            logging.warning(f"Configured model '{GEMINI_MODEL_NAME}' is not listed. Trying anyway.")
        model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
        logging.info(f"Initialized Gemini model: {GEMINI_MODEL_NAME}")
except Exception as e:
    logging.error(f"Failed to initialize Gemini model: {e}")

def transcribe_audio_vosk(recognizer_instance, audio_source):
    logging.info("Listening for your command (Vosk offline)...")
    rec = KaldiRecognizer(vosk_model, 16000)
    rec.SetWords(False)

    with audio_source as source:
        recognizer_instance.adjust_for_ambient_noise(source, duration=0.5)
        def audio_stream():
            try:
                audio_data_raw = recognizer_instance.listen(source, phrase_time_limit=8, timeout=5, chunk_size=1024)
                for chunk in audio_data_raw.get_raw_data(convert_rate=16000, convert_width=2):
                    yield chunk
            except sr.WaitTimeoutError:
                logging.warning("No speech detected.")
                yield b""
            except Exception as e:
                logging.error(f"Error during audio capture: {e}")
                yield b""

        full_text = ""
        for audio_chunk in audio_stream():
            if not audio_chunk:
                break
            if rec.AcceptWaveform(audio_chunk):
                result_json = json.loads(rec.Result())
                if 'text' in result_json:
                    full_text += result_json['text'] + " "
        final_result_json = json.loads(rec.FinalResult())
        if 'text' in final_result_json:
            full_text += final_result_json['text']
        full_text = full_text.strip()
        logging.info(f"Transcribed: '{full_text}'")
        return full_text if full_text else "Sorry, I could not understand the audio."

def generate_gpt_response(prompt):
    clean_prompt = prompt.lower().strip()
    if clean_prompt in ["who created you", "who is your creator"]:
        return "I was created by Anandu Murali."
    if clean_prompt == "what is metamind":
        return "METAMIND is an AI assistant built with Django, Gemini AI, and Vosk."
    if clean_prompt == "stop":
        return "Ok"
    if clean_prompt in ["how am i feeling", "tell me my emotion", "what's my emotional state", "what is my current emotion"]:
        return f"Based on facial analysis, you appear to be feeling {current_emotion_state}. "

    # --- Desktop App Triggers (Windows) ---
    if clean_prompt.startswith("open "):
        app_name = clean_prompt.replace("open ", "").strip()
        known_apps = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "chrome": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            "command prompt": "cmd.exe"
            
        }
        if app_name in known_apps:
            try:
                subprocess.Popen(known_apps[app_name])
                return f"Opening {app_name}..."
            except Exception as e:
                return f"Failed to open {app_name}: {str(e)}"

    if not prompt or clean_prompt in ["no speech detected.", "sorry, i could not understand the audio.", "offline recognition error."]:
        return "Please tell me how can I help you."

    if not gemini_api_key:
        return "I can only process simple commands in offline mode."

    try:
        model_instance = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
        response = model_instance.generate_content(prompt)
        return response.text.strip()
    except genai.types.BlockedPromptException:
        return "I'm sorry, I cannot respond to that query."
    except Exception as e:
        return f"Error fetching Gemini response: {str(e)}"

def generate_greeting(emotion):
    prompt = f"Generate a short greeting for a user who seems to be feeling {emotion}. Make it friendly and empathetic."
    return generate_gpt_response(prompt)

def text_to_speech_pyttsx3(text):
    if not text:
        return None
    try:
        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        tts_engine.stop()
        tts_engine.save_to_file(text, audio_path)
        tts_engine.runAndWait()
        return audio_path
    except Exception as e:
        logging.error(f"TTS error: {e}")
        return None

def analyze_emotion_and_identity(frame):
    global current_emotion_state
    if frame is None or not isinstance(frame, np.ndarray):
        return {"emotion": "unknown", "error": "Invalid frame input."}

    fd, temp_img_path = tempfile.mkstemp(suffix=".jpg")
    try:
        cv2.imwrite(temp_img_path, frame)
        os.close(fd)

        result = DeepFace.analyze(img_path=temp_img_path, actions=['emotion'], enforce_detection=False)
        emotion = result[0].get("dominant_emotion", "neutral")
        current_emotion_state = emotion
        greeting = generate_greeting(emotion)
        logging.info(f"Greeting based on emotion '{emotion}': {greeting}")
        audio_path = text_to_speech_pyttsx3(greeting)

        return {
            "emotion": emotion,
            "greeting": greeting,
            "greeting_audio_path": audio_path
        }
    except Exception as e:
        return {"emotion": "neutral", "error": str(e)}
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
