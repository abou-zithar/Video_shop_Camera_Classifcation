
from django.shortcuts import render
from .forms import VideoUploadForm
from utils import classify_video

# View for video upload and classification
def video_upload_view(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_instance = form.save()
            video_path = video_instance.video.path

            # Classify the video
            prediction, accuracy = classify_video(video_path)

            return render(request, 'classify/success.html', {'prediction': prediction, 'accuracy': accuracy})
    else:
        form = VideoUploadForm()

    return render(request, 'classify/upload.html', {'form': form})

# Home view
def home_view(request):
    return render(request, 'classify/home.html')
