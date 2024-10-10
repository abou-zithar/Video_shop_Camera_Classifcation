# classify/urls.py
from django.urls import path
from .views import home_view, video_upload_view

urlpatterns = [
    path('', home_view, name='home'),  # Home page
    path('upload/', video_upload_view, name='upload'),  # Upload page
]
