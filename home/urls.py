from django.urls import path, include
from . import views
urlpatterns = [
    path('',views.index ,name='home'),
    path('about/',views.about,name='about'),
    path('contact/',views.contact,name='contact'),
    path('metamind/',views.metamind,name='metamind'),
    path('services/',views.services,name='services'),
]
