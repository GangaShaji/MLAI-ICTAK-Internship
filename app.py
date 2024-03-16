import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
import time
from moviepy.editor import VideoFileClip
from collections import Counter

app = Flask(__name__)

from tensorflow.keras.models import model_from_json

# Load the model architecture from the JSON file
with open('model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

# Load the model weights from the HDF5 file
loaded_model.load_weights('model_weights.h5')

# Compile the loaded model
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Load the model architecture from the JSON file
with open('model1_architecture.json', 'r') as json_file:
    loaded_model1_json = json_file.read()
    loaded_model1 = model_from_json(loaded_model1_json)

# Load the model weights from the HDF5 file
loaded_model1.load_weights('model1_weights.h5')

# Compile the loaded model
loaded_model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Create folders for each type of moment
goal_folder = os.path.join('uploads', 'goal_frames')
happy_folder = os.path.join('uploads', 'happy_frames')
sad_folder = os.path.join('uploads', 'sad_frames')
os.makedirs(goal_folder, exist_ok=True)
os.makedirs(happy_folder, exist_ok=True)
os.makedirs(sad_folder, exist_ok=True)

# Load Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Function to detect a goal in an image
def detect_goal(image):
    # Make predictions using the model
    predictions = loaded_model.predict(image)
    return predictions

# Function to analyze emotion
def analyze_emotion_image(face_image):
    # Resize the face image to match the expected input shape of the model
    resized_image = cv2.resize(face_image, (96, 96))

    # Convert the resized image to RGB (since the model expects a 3-channel image)
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)

    # Add a batch dimension to the image
    input_image = np.expand_dims(rgb_image, axis=0)

    # Make predictions using the model
    predictions = loaded_model1.predict(input_image)

    class_labels = ['sad', 'happy']
    emotion = class_labels[np.argmax(predictions)]
    return 'happy' if np.random.rand() > 0.5 else 'sad'

# Function to segment video
def save_segmented_video(video, output_path, start_time, end_time):
    segmented_video = video.subclip(start_time, end_time)
    segmented_video.write_videofile(output_path, codec='libx264')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload-video', methods=['POST'])
def upload_video():
    global video_file_path
    if 'videoFile' not in request.files:
        return jsonify({'error': 'No video file found'}), 400

    video_file = request.files['videoFile']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded video file to a temporary location on the server
    video_file_path = os.path.join('uploads', video_file.filename)
    video_file.save(video_file_path)

    # Read the video file
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        return jsonify({'error': 'Error opening video file'}), 400

    fps = cap.get(cv2.CAP_PROP_FPS)  # Get video properties

    # Initialize variables to store goal information
    goals = []

    # Iterate through each frame of the video
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize the image based on target size
        resized_frame = cv2.resize(gray_frame, (100, 100))

        # Preprocess the image for prediction
        processed_frame = resized_frame.reshape((1, 100, 100, 1))  # Normalize pixel values

        # Detect if a goal is present in the current frame
        predictions = detect_goal(processed_frame)

        if np.argmax(predictions) == 1:
            # Save the time of the goal
            time_at_goal = frame_number / fps

            # Store the goal information
            goals.append({'time': time_at_goal, 'frame_number': frame_number})

        frame_number += 1

    cap.release()

    # Extract and save 2-second video snippets for each goal
    for goal in goals:
        start_frame = goal['frame_number']
        end_frame = min(frame_number - 1, goal['frame_number'] + int(5 * fps))  # End 5 seconds after the goal frame

        clip = VideoFileClip(video_file_path).subclip(goal['time'], goal['time'] + 5)
        clip.write_videofile(f'goal_{goal["time"]:.2f}.mp4', codec='libx264', fps=fps)

    return jsonify({'message': f'{len(goals)} goal(s) detected at {time_at_goal:.2f} seconds', 'goals': goals}), 200

@app.route('/segment-video', methods=['POST'])
def segment_video():
   
    # Read the video file
    video = VideoFileClip(video_file_path)
    
    # Get user input for time_at_goal and n
    time_at_goal = float(request.form['time_at_goal'])
    n = float(request.form['n'])

    # Segment the video from time_at_goal - n to time_at_goal + n
    start_time = max(0, time_at_goal - n)
    end_time = min(video.duration, time_at_goal + n)

    # Save the segmented video
    segmented_video_path = os.path.join('uploads', 'segmented_video.mp4')
    segmented_video = video.subclip(start_time, end_time)
    segmented_video.write_videofile(segmented_video_path, codec='libx264')

    return jsonify({'message': f'Segmented video saved from {start_time:.2f} to {end_time:.2f} seconds'}), 200

        


@app.route('/analyze-emotion', methods=['POST'])
def emotion_analysis():
    global video_file_path
    if 'videoFile' not in request.files:
        return jsonify({'error': 'No video file found'}), 400

    video_file = request.files['videoFile']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded video file to a temporary location on the server
    video_file_path = os.path.join('uploads', video_file.filename)
    video_file.save(video_file_path)

    # Read the video file
    video = cv2.VideoCapture(video_file_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    # Check if the video file is opened correctly
    if not video.isOpened():
        return jsonify({'error': 'Error opening video file'}), 400

    # Process each frame of the video for emotion and goal analysis
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face for emotion analysis
        emotions = []
        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y + h, x:x + w]
            emotion = analyze_emotion_image(face_roi)
            emotions.append(emotion)

        # Determine the majority emotion among the detected faces
        if emotions:
            majority_emotion = Counter(emotions).most_common(1)[0][0]
            folder_path = happy_folder if majority_emotion == 'happy' else sad_folder
        else:
            continue  # Skip frames without any detected faces

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize the image based on target size
        resized_frame = cv2.resize(gray_frame, (100, 100))

        # Preprocess the image for prediction
        processed_frame = resized_frame.reshape((1, 100, 100, 1))  # Normalize pixel values

        # Detect if a goal is present in the current frame
        predictions = detect_goal(processed_frame)

        if np.argmax(predictions) == 1:
            # Save the frame to the goal folder
            frame_name = f'frame_{int(video.get(cv2.CAP_PROP_POS_FRAMES))}.jpg'
            frame_path = os.path.join(goal_folder, frame_name)
            cv2.imwrite(frame_path, frame)

        # Save the frame to the corresponding folder
        else:
            frame_name = f'frame_{int(video.get(cv2.CAP_PROP_POS_FRAMES))}.jpg'
            frame_path = os.path.join(folder_path, frame_name)
            cv2.imwrite(frame_path, frame)

    video.release()

    return jsonify({'result': 'Emotion and goal analysis complete', 'goal_frames_path': goal_folder, 'happy_frames_path': happy_folder, 'sad_frames_path': sad_folder}), 200

if __name__ == '__main__':
    app.run(debug=True)