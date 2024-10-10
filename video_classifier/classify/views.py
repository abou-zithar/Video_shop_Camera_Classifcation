import os
import cv2
import torch
from django.conf import settings
from django.shortcuts import render
from .forms import VideoUploadForm
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

# Define device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the processor and model
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics",
    num_labels=2,  # The number of output classes
    ignore_mismatched_sizes=True  # Ignore size mismatches in the classifier layer
).to(device)

# Load the weights from your .pth file
MODEL_PATH = os.path.join(settings.BASE_DIR, 'model_pre_trained.pth')
state_dict = torch.load(MODEL_PATH, map_location=device)  # Load state dict from .pth file
model.load_state_dict(state_dict, strict=False)  # Load the state dictionary with strict=False

# Label mapping
label_mapping = {0: "Thief", 1: "Not a Thief"}

# Function to extract 16 evenly spaced frames from the video
def extract_frames(video_path, num_frames=16, height=224, width=224):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        print(f"Video has fewer frames than requested ({total_frames} < {num_frames}).")
        return []

    frame_interval = total_frames // num_frames
    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            frames.append(frame)
        else:
            break

    cap.release()
    return frames

# Preprocess the frames
def preprocess_video_frames(frames):
    if not frames:
        return None
    inputs = processor(frames, return_tensors="pt")
    return inputs

# Classify the video frames
def classify_video(video_path):
    try:
        # Extract 16 frames from the video
        frames = extract_frames(video_path, num_frames=16)
        if not frames:
            return "Error: No frames extracted.", 0

        # Preprocess the frames
        inputs = preprocess_video_frames(frames)
        if inputs is None:
            return "Error: Preprocessing failed.", 0

        # Move the preprocessed frames to the device
        pixel_values = inputs['pixel_values'].to(device)

        # Set the model to evaluation mode
        model.eval()

        # Run inference without computing gradients
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits

        # Get the predicted class
        predicted_class_idx = torch.argmax(logits, dim=1).item()
        predicted_class = label_mapping.get(predicted_class_idx, "Unknown")

        # Calculate accuracy (for demonstration, set to 100% for the predicted class)
        accuracy = 100  # This should be based on your model's prediction confidence if available

        return predicted_class, accuracy

    except ValueError as e:
        print(f"ValueError during classification: {e}")
        return "ValueError during classification", 0
    except Exception as e:
        print(f"Error during classification: {e}")
        return "Error during classification", 0

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
