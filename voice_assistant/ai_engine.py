
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
import pyautogui
import pyperclip
import re
import requests
from datetime import datetime

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

current_emotion_state = "neutral"
last_interaction_time = time.time()

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

# OpenWeatherMap API Key
WEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
if not WEATHER_API_KEY:
    logging.warning("OPENWEATHER_API_KEY not set in .env file.")

LOCATION = "Mangaluru, Karnataka, India" # Default location, can be made dynamic later

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

def send_whatsapp_message(contact_name, message):
    try:
        subprocess.Popen(['whatsapp']) # Try to open WhatsApp directly
        time.sleep(5) # Give it time to open

        pyautogui.hotkey('ctrl', 'f')
        time.sleep(1)
        pyperclip.copy(contact_name)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(2)
        pyautogui.press('down')
        pyautogui.press('enter')
        time.sleep(1)

        pyperclip.copy(message)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(1)
        pyautogui.press('enter')

        return f"Message sent to {contact_name}."
    except Exception as e:
        return f"Failed to send WhatsApp message: {str(e)}"

def play_song_on_spotify(song_name):
    try:
        # Instead of win+r, try directly opening spotify if it's in PATH or a known location
        subprocess.Popen(['spotify'])
        time.sleep(7)

        pyautogui.hotkey('ctrl', 'k') # Focus search in Spotify
        time.sleep(1)
        pyperclip.copy(song_name)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(2)
        pyautogui.press('enter')

        return f"Playing '{song_name}' on Spotify."
    except Exception as e:
        return f"Failed to play song: {str(e)}"
def stop_playing():
    pyautogui.hotkey('win', 'r')
    time.sleep(1)
    pyperclip.copy('spotify')
    pyautogui.hotkey('ctrl', 'v')
    pyautogui.press('enter')
    time.sleep(5)
    pyautogui.press('space')
    return "Ok, music stopped playing."
def resume_playing():
    pyautogui.hotkey('win', 'r')
    time.sleep(1)
    pyperclip.copy('spotify')
    pyautogui.hotkey('ctrl', 'v')
    pyautogui.press('enter')
    time.sleep(5)
    pyautogui.press('space')
    return "Ok, music resumed."

def get_emotion_based_response(emotion):
    try:
        if emotion == "sad":
            prompt = "suggest a joke for a user in sad mood.just tell me joke only no other sentences no need"
            model_instance = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
            result = model_instance.generate_content(prompt).text.strip()
            return "joke", result
        else:
            # For all other emotions, suggest a song
            prompt = f"suggest a song for a user in {emotion} mood.need only one song.i need to play it in spotify so only song name is needed"
            model_instance = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
            result = model_instance.generate_content(prompt).text.strip()
            # Assuming the Gemini model directly returns the song name for this prompt
            return "song", result
    except Exception as e:
        logging.error(f"Gemini emotion response error: {e}")
    return None, None

def get_intro_line(emotion):
    try:
        prompt = f"User seems {emotion}. Generate a friendly, empathetic intro line before suggesting a joke or song."
        return generate_gpt_response(prompt)
    except Exception as e:
        logging.error(f"Gemini intro line error: {e}")
        return "Hey, let me cheer you up a bit!"

def silent_mode_watcher():
    global last_interaction_time
    while True:
        time.sleep(10)
        if assistant_active.is_set():
            continue
        if time.time() - last_interaction_time > 60:
            intro = get_intro_line(current_emotion_state)
            response_type, response_content = get_emotion_based_response(current_emotion_state)

            if intro:
                text_to_speech_pyttsx3(intro)

            if response_type == "joke":
                text_to_speech_pyttsx3(response_content)
            elif response_type == "song":
                play_song_on_spotify(response_content)
            else:
                logging.warning("Unexpected response format or empty response from emotion-based response.")

            last_interaction_time = time.time()


def get_current_time():
    now = datetime.now()
    return now.strftime("The current time is %I:%M %p.")

def get_current_weather(location):
    if not WEATHER_API_KEY:
        return "I cannot fetch weather information, my weather API key is not configured."

    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    city_name = location.split(',')[0].strip() # Extract city from location string
    complete_url = f"{base_url}q={city_name}&appid={WEATHER_API_KEY}&units=metric"
    
    try:
        response = requests.get(complete_url)
        data = response.json()

        if data["cod"] != "404":
            main = data["main"]
            weather = data["weather"][0]
            temperature = main["temp"]
            description = weather["description"]
            return f"The weather in {city_name} is currently {description} with a temperature of {temperature:.1f} degrees Celsius."
        else:
            return "I could not find weather information for that location."
    except Exception as e:
        return f"Failed to get weather information: {str(e)}"


def generate_gpt_response(prompt):
    global last_interaction_time
    last_interaction_time = time.time()

    clean_prompt = prompt.lower().strip()
    
    if clean_prompt in ["who created you", "who is your creator"]:
        return "I was created by Anandu Murali."
    if clean_prompt == "what is metamind":
        return "METAMIND is an AI assistant built with Django, Gemini AI, and Vosk."
    if clean_prompt == "stop":
        return "Ok"
    if clean_prompt in ["how am i feeling", "tell me my emotion", "what's my emotional state", "what is my current emotion"]:
        return f"Based on facial analysis, you appear to be feeling {current_emotion_state}. "
    
    if clean_prompt == "what time is it":
        return get_current_time()
    
    if clean_prompt == "what's the weather like" or clean_prompt == "tell me about the current weather":
        return get_current_weather(LOCATION)
    if clean_prompt == "stop playing song" or clean_prompt =="pause the song" :
        return stop_playing()
    if clean_prompt == "resume" or clean_prompt =="continue the song" :
        return resume_playing()
    match = re.match(r"send whatsapp message to (.+?) say (.+)", clean_prompt)
    if match:
        contact_name = match.group(1).strip()
        message = match.group(2).strip()
        return send_whatsapp_message(contact_name, message)

    match_song = re.match(r"play (.+)", clean_prompt)
    if match_song:
        song_name = match_song.group(1).strip()
        return play_song_on_spotify(song_name)

    if clean_prompt.startswith("open "):
        app_name = clean_prompt.replace("open ", "").strip()
        known_apps = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "chrome": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", # Ensure correct path
            "command prompt": "cmd.exe",
            "whatsapp": "whatsapp.exe", # Added for direct open
            "spotify": "spotify.exe" # Added for direct open
        }
        if app_name in known_apps:
            try:
                subprocess.Popen(known_apps[app_name])
                return f"Opening {app_name}..."
            except Exception as e:
                return f"Failed to open {app_name}: {str(e)}. Please ensure the application is correctly installed and its path is accessible."
        else:
            return f"I don't know how to open {app_name}. Can you tell me its full path or if it's a standard application?"

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

# Start background silence detection
watcher_thread = threading.Thread(target=silent_mode_watcher, daemon=True)
watcher_thread.start()
