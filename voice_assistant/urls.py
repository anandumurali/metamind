# metamind/urls.py

from django.urls import path
from . import views

app_name = 'metamind'

urlpatterns = [
    path('', views.metamind, name='metamind'),
    path('chat/', views.chat_api, name='chat_api'),
    path('emotion/', views.emotion_api, name='emotion_api'),
]
