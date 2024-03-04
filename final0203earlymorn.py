import gradio as gr
import cv2
import dlib
import shutil
import numpy as np
import random
from datetime import datetime
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import warnings
import glob
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
from PIL import Image
from PIL.ExifTags import TAGS
import tempfile
import librosa
import plotly.express as px
import torchaudio
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead

warnings.filterwarnings("ignore")

def inputseparation(video, image, audio):
    if video is not None:
        return save_video(video)
    elif image is not None:
        return predictimage(image)
    else:
        return audiopredict(audio)

def load_audio(uploaded_file, sampling_rate=22000):
    
    # Handle MP3 files with torchaudio
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        with open(uploaded_file, 'rb') as audio_file:  # Open in binary mode
                tmp.write(audio_file.read())
        tmp_path = tmp.name
        audio, sr = torchaudio.load(tmp_path)
        audio = audio.mean(dim=0)

    if sr != sampling_rate:
        audio = torchaudio.transforms.Resample(sr, sampling_rate)(audio)

    audio = audio.clamp_(-1, 1)

    return audio.unsqueeze(0)


def classify_audio_clip(clip):
    classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4, resnet_blocks=2, attn_blocks=4, num_attn_heads=4, base_channels=32, dropout=0, kernel_size=5, distribute_zero_label=False)
    state_dict = torch.load('classifier.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(state_dict)
    classifier.eval()
    clip = clip.cpu().unsqueeze(0)    
    with torch.no_grad():
        results = classifier(clip)
        probabilities = F.softmax(results, dim=-1)
    ai_generated_probability = probabilities[0][1].item()
    return ai_generated_probability

def audiopredict(audio):
    if audio is not None:
        audio_clip = load_audio(audio)
        ai_generated_probability = classify_audio_clip(audio_clip)
        image_path = os.path.join("D:\\Python ML\\cyberhackathon\\mainfolder\\audiobg.jpg")
        image = Image.open(image_path)
        if ai_generated_probability < 0.5:
            return "Real", "The audio is likely to be Real", "No EXIF data found in the audio", image
        else:
            return "Deepfake", "The audio is likely to be AI Generated", "No EXIF data found in the audio", image
    
# Video Input Code
def save_video(video_path):
    
    # Modify this path to your desired folder
    save_folder = "D:\\Python ML\\cyberhackathon\\mainfolder\\videos"

    # Extract filename from path
    filename = video_path.split("\\")[-1]

    # Save video to specified folder
    with open(f"{save_folder}/{filename}", "wb") as f:
        f.write(open(video_path, "rb").read())

    # Process frames, select faces, and perform deepfake identification
    textoutput, exif, face_with_mask = process_video(save_folder, filename)   
    print(textoutput)
    string = textoutput

    # Extract percentages and convert them to floats
    percentages = re.findall(r"(\d+\.\d+)%", string)
    real_percentage = float(percentages[0])
    fake_percentage = float(percentages[1])

    # Determine which percentage is higher
    if real_percentage > fake_percentage:
        print("Real")
        val = "Real"
    else:
        print("Fake")
        val = "Deepfake"

    
    return val, textoutput, exif, face_with_mask

def process_video(video_folder, video_filename):
    # Additional Processing (Frames, Faces, Deepfake Identification)
    frames_base_dir = "D:\\Python ML\\cyberhackathon\\mainfolder\\frames"
    faces_base_dir = "D:\\Python ML\\cyberhackathon\\mainfolder\\faces"
    selected_faces_base_dir = "D:\\Python ML\\cyberhackathon\\mainfolder\\selectedfaces"

    # Find the latest video
    video_path = os.path.join(video_folder, video_filename)
    
    # Create session folders
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_session_dir = create_session_folder(frames_base_dir, session_name)
    faces_session_dir = create_session_folder(faces_base_dir, session_name)
    selected_faces_session_dir = create_session_folder(selected_faces_base_dir, session_name)

    # Extract frames and faces
    video_to_frames_and_extract_faces(video_path, frames_session_dir, faces_session_dir)
    
    # Select random faces
    select_random_faces(faces_session_dir, selected_faces_session_dir)

    # Perform deepfake identification
    textoutput, exif, face_with_mask = identify_deepfake(selected_faces_session_dir)
    return textoutput, exif, face_with_mask

def create_session_folder(parent_dir, session_name=None):
    if not session_name:
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_path = os.path.join(parent_dir, session_name)
    os.makedirs(session_path, exist_ok=True)
    return session_path

def extract_faces(frame_path, faces_dir):
    frame = cv2.imread(frame_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    faces = detector(gray, 1)

    faces_extracted = 0
    for (i, face) in enumerate(faces):
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        face_image = frame[y:y+h, x:x+w]
        face_file_path = os.path.join(faces_dir, f"face_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
        cv2.imwrite(face_file_path, face_image)
        faces_extracted += 1

    return faces_extracted

def video_to_frames_and_extract_faces(video_path, frames_dir, faces_dir):
    video_capture = cv2.VideoCapture(video_path)
    success, frame = video_capture.read()
    frame_count = 0
    processed_frame_count = 0  
    futures = []
   
    num_workers = min(multiprocessing.cpu_count(), 8)  

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        while success:
           
            if frame_count % 2 == 0:
                frame_file = os.path.join(frames_dir, f"frame_{processed_frame_count}.jpg")
                cv2.imwrite(frame_file, frame)
                processed_frame_count += 1

                
                if processed_frame_count % 4 == 0:
                    future = executor.submit(extract_faces, frame_file, faces_dir)
                    futures.append(future)

            success, frame = video_capture.read()
            frame_count += 1

    total_faces = sum(f.result() for f in as_completed(futures))
    print(f"Saved frames: {processed_frame_count}, Processed for face extraction: {len(futures)}, Extracted faces: {total_faces}")

    video_capture.release()
    return total_faces

def select_random_faces(faces_dir, selected_faces_dir):
    face_files = [os.path.join(faces_dir, f) for f in os.listdir(faces_dir) if f.endswith('.jpg')]
    selected_faces = random.sample(face_files, min(20, len(face_files)))  
    for face_file in selected_faces:
        basename = os.path.basename(face_file)
        destination_file = os.path.join(selected_faces_dir, basename)
        shutil.copy(face_file, destination_file)  

    print(f"Selected random faces: {len(selected_faces)}")

# Find Deepfake or Not
def identify_deepfake(selected_faces_dir):
    # Setup device
    DEVICE = 'cpu' if not torch.cuda.is_available() else 'cuda'

    # Initialize MTCNN and InceptionResnetV1 with pre-trained models
    mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).to(DEVICE).eval()
    model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=1, device=DEVICE)

    # Load the model checkpoint
    checkpoint_path = "resnetinceptionv1_epoch_32.pth"  # Update this path
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    # Define prediction function
    def predict(input_image: Image.Image):
        try:
            face = mtcnn(input_image)
            if face is None:
                raise Exception('No face detected')
            
            face = F.interpolate(face.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
            face = face.to(DEVICE).to(torch.float32) / 255.0

            target_layers = [model.block8.branch1[-1]]
            cam = GradCAM(model=model, target_layers=target_layers)
            targets = [ClassifierOutputTarget(0)]

            grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
            grayscale_cam = grayscale_cam[0, :]
            face_image_np = face.squeeze().permute(1, 2, 0).cpu().detach().numpy()
            visualization = show_cam_on_image(face_image_np, grayscale_cam, use_rgb=True)
            face_with_mask = cv2.addWeighted((face_image_np * 255).astype('uint8'), 1, (visualization * 255).astype('uint8'), 0.5, 0)
            
            with torch.no_grad():
                output = torch.sigmoid(model(face)).item()
                prediction = "real" if output < 0.5 else "fake"
                confidences = {'real': 1 - output, 'fake': output}
            
            return confidences, prediction, face_with_mask

        except Exception as e:
            print(f"Prediction failed: {e}")
            return {'real': 0, 'fake': 100}, "fake", None

    # Process images in the selected folder
    image_files = sorted([f for f in os.listdir(selected_faces_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    results = {}  # Initialize an empty dictionary to store results

    for image_file in image_files:
        image_path = os.path.join(selected_faces_dir, image_file)
        input_image = Image.open(image_path)
        
        confidences, prediction, face_with_mask = predict(input_image)   
        # print(confidences, prediction, face_with_mask) 
        if face_with_mask is None:
            continue
        
        # Store the results in the dictionary
        results[image_file] = {
            'Confidence': confidences,
            'Prediction': 'real' if confidences['real'] > confidences['fake'] else 'fake'
        }
        print(f"Image: {image_file}, Confidence: {confidences}, Prediction: {'real' if confidences['real'] > confidences['fake'] else 'fake'}")
    
    image_path = os.path.join(selected_faces_dir, image_files[0])
    image = Image.open(image_path)
    exif_data = image.getexif()  # Returns an Exif instance or None

    if exif_data:
        exif = ""
        for tag_id in exif_data:
            # Get the tag name
            tag = TAGS.get(tag_id, tag_id)
            value = exif_data[tag_id]
            # Print the tag and value in a human-readable format
            exif += f"{tag}: {value}\n"
    else:
        exif = "No EXIF data or Metadata found in the video"
    
    # Accumulate 'real' and 'fake' scores
    real_total = 0.0
    fake_total = 0.0
    count = 0  

    for key, value in results.items():
        if 'Confidence' in value:
            real_total += value['Confidence']['real']
            fake_total += value['Confidence']['fake']
            count += 1

    # Calculate and display consolidated score if any images were successfully processed
    if count > 0:
        real_avg = (real_total / count) * 100
        fake_avg = (fake_total / count) * 100
        
        textoutput = (f"Consolidated Score for the uploaded video - Real: {real_avg:.2f}%, Fake: {fake_avg:.2f}%")
        
        return textoutput, exif, face_with_mask
        
    else:
        print("No images were successfully processed to calculate a consolidated score.")

# Gradio Interface
def predictimage(input_image: Image.Image):
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    mtcnn = MTCNN(
        select_largest=False,
        post_process=False,
        device=DEVICE
    ).to(DEVICE).eval()

    model = InceptionResnetV1(
        pretrained="vggface2",
        classify=True,
        num_classes=1,
        device=DEVICE
    )

    checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    face = mtcnn(input_image)
    image = input_image
    exif_data = image.getexif()  # Returns an Exif instance or None

    if exif_data:
        exif = ""
        for tag_id in exif_data:
            # Get the tag name
            tag = TAGS.get(tag_id, tag_id)
            value = exif_data[tag_id]
            # Print the tag and value in a human-readable format
            exif += f"{tag}: {value}\n"
    else:
        exif = "No EXIF data found in the image"
    if face is None:
        return "Neutral", "No face detected", exif, input_image
    face = face.unsqueeze(0) # add the batch dimension
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
    
    # convert the face into a numpy array to be able to plot it
    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    prev_face = prev_face.astype('uint8')

    face = face.to(DEVICE)
    face = face.to(torch.float32)
    face = face / 255.0
    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

    target_layers=[model.block8.branch1[-1]]
    use_cuda = True if torch.cuda.is_available() else False
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)]

    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "Real" if output.item() < 0.5 else "Deepfake"
        
        real_prediction = 1 - output.item()
        fake_prediction = output.item()
        
        real_avg = real_prediction * 100
        fake_avg = fake_prediction * 100
        
        textoutput = (f"Consolidated Score for the uploaded image - Real: {real_avg:.2f}%, Fake: {fake_avg:.2f}%")
        
                
    return prediction, textoutput, exif, face_with_mask

def main():
    # Video Input Interface
    video_input_interface = gr.Interface(
        fn=inputseparation,
        inputs=[
            gr.Video(label="Upload Video"),
            gr.Image(label="Input Image", type="pil"),
            gr.Audio(label="Upload Audio", type="filepath")
        ],
        outputs=[
            gr.Label(label="Output Result"),
            gr.Text(label="Explanation"),
            gr.Text(label="EXIF Data / Metadata"),  
            gr.Image(label="Face with Mask")         
        ],
        title="Veritrue.ai",
        description="You can upload either a video, image or an audio and it will give you whether it is a deepfake or a real one."
    )

    # Execute Video Input Interface
    video_input_interface.launch()

if __name__ == "__main__":
    main()
