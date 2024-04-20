import matplotlib
import os

matplotlib.use('Agg')

import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
import cv2
from skimage.transform import resize
import time
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import tensorflow as tf
from model_functions import mamon_videoFightModel2, video_mamonreader, pred_fight, main_fight

app = Flask(__name__)

reports = []

REPORTS_FOLDER = 'reports'

if not os.path.exists(REPORTS_FOLDER):
    os.makedirs(REPORTS_FOLDER)
# Load your TensorFlow model
np.random.seed(1234)
model22 = mamon_videoFightModel2(tf)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST' and 'file' in request.files:
        # Get the uploaded video file
        file = request.files['file']
        
        # Save the video temporarily
        video_path = './uploaded_video.mp4'
        file.save(video_path)
        
        # Perform analysis on the video
        res = main_fight(video_path, model22)
        
        # Release the resources associated with the video file
        file.close()  # Close the file handle
        cv2.VideoCapture(video_path).release()  # Release the video capture object
        
        # Remove the video file
        os.remove(video_path)

        confidence = float(res['precentegeoffight']) * 100  # Convert confidence to percentage
        plt.figure(figsize=(6, 4))
        plt.bar(['Confidence Level'], [confidence], color='skyblue')
        plt.ylabel('Percentage (%)')
        plt.title('Confidence Level of Violence Detection')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('./static/confidence_plot.png')  # Save the plot as an image
        plt.close()
        
        # Return the analysis result as JSON
        return jsonify(res)
    return jsonify({'error': 'Invalid request'})

@app.route('/report.html', methods=['GET', 'POST'])
def report():
    response = None
    if request.method == 'POST':
        name = request.form.get('name')
        # Handle the feedback or error report sent by the user
        # For example, you can access the form data using request.form
        feedback = request.form.get('feedback')

        reports.append({'name': name, 'feedback': feedback})

        filename = os.path.join(REPORTS_FOLDER, f"{name}_report.txt")
        with open(filename, 'w') as file:
            file.write(feedback)
        # Process the feedback as needed
       # Store the report along with the user's name
        
        response = {'message' : 'Report submitted successfully!'}

    # Render the report page template
    return render_template('report.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)
