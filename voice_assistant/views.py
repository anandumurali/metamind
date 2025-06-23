# voice_assistant/views.py

import base64
import cv2
import numpy as np
import os

from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json

from .ai_engine import generate_gpt_response, text_to_speech_pyttsx3, analyze_emotion_and_identity

def metamind(request):
    """
    Renders the main HTML page for the MetaMind assistant.
    """
    return render(request, 'metamind/voice_assistant/metamind.html')


@csrf_exempt
@require_http_methods(["POST"])
def chat_api(request):
    """
    Handles user text input (from typed or voice command),
    generates AI response, and TTS audio (base64 encoded).
    """
    try:
        data = json.loads(request.body.decode())
        user_input = data.get('message', '')
        if not user_input:
            return JsonResponse({'error': 'Empty message'}, status=400)

        bot_reply = generate_gpt_response(user_input)
        audio_path = text_to_speech_pyttsx3(bot_reply)

        if not audio_path or not os.path.exists(audio_path):
            return JsonResponse({'error': 'Failed to generate audio'}, status=500)

        with open(audio_path, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode()

        os.remove(audio_path)

        return JsonResponse({
            'response': bot_reply,
            'audio': audio_data
        })

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@csrf_exempt
def emotion_api(request):
    if request.method == 'POST':
        image_data = request.POST.get('frame', '')
        if not image_data:
            return JsonResponse({'error': 'No image data provided'}, status=400)

        try:
            encoded_data = image_data.split(',')[1]
            np_data = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            result = analyze_emotion_and_identity(frame)

            greeting = result.get('greeting')
            audio_path = result.get('greeting_audio_path')
            audio_base64 = None

            if audio_path and os.path.exists(audio_path):
                with open(audio_path, 'rb') as f:
                    audio_base64 = base64.b64encode(f.read()).decode()
                os.remove(audio_path)

            return JsonResponse({
                'emotion': result.get('emotion', 'neutral'),
                'greeting': greeting,
                'audio': audio_base64
            })

        except Exception as e:
            return JsonResponse({'error': f'Failed to process image: {str(e)}'}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=405)
