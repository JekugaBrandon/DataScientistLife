import cv2
import numpy as np
import time
import os
import json
import sqlite3
import datetime
import csv
from deepface import DeepFace
import dlib
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import logging
from collections import deque
import pyaudio
import librosa
import threading
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import winsound
from abc import ABC, abstractmethod
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import boto3
from botocore.exceptions import NoCredentialsError
from google.cloud import storage
from cryptography.fernet import Fernet
from twilio.rest import Client
import firebase_admin
from firebase_admin import credentials, messaging
import speech_recognition as sr
import pyttsx3
from googletrans import Translator
import uuid
import requests
import zipfile
import io
import platform
import subprocess
import sys
import getpass
import nltk
from nltk.tokenize import word_tokenize
from pynput import keyboard, mouse

# Suppress TensorFlow warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

nltk.download('punkt')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='interaction_analysis.log')

# ConfigManager
class ConfigManager:
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.default_config = {
            'models_dir': 'models',
            'known_faces_dir': 'known_faces',
            'db_path': 'interaction_analysis.db',
            'csv_path': 'interaction_data.csv',
            'anxiety_threshold': 0.65,
            'confidence_threshold': 0.5,
            'alert_cooldown': 10,
            'frame_skip': 2,
            'email_sender': 'your_email@gmail.com',
            'email_password': 'your_app_password',
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'dash_port': 8050,
            'chatbot_model': 'microsoft/DialoGPT-medium',
            'cloud_storage': 'local',
            'aws_access_key': '',
            'aws_secret_key': '',
            'gcp_credentials': '',
            'encryption_key': Fernet.generate_key().decode(),
            'twilio_sid': '',
            'twilio_token': '',
            'twilio_phone': '',
            'firebase_credentials': '',
            'voice_assistant': 'off',
            'language': 'en',
            'offline_mode': False,
            'api_enabled': False,
            'api_port': 5000,
            'gamification': True,
            'social_features': True,
            'accessibility': True,
            'user_profile': {'name': '', 'age': 0, 'goals': {}, 'preferred_techniques': []}
        }
        self.config = self.load_config()

    def load_config(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            else:
                with open(self.config_file, 'w') as f:
                    json.dump(self.default_config, f, indent=4)
                return self.default_config
        except Exception as e:
            logging.error(f"Config load error: {e}")
            return self.default_config

# UserProfileManager
class UserProfileManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self._initialize_profile_db()

    def _initialize_profile_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                age INTEGER,
                goals TEXT,
                preferred_techniques TEXT,
                anxiety_threshold REAL
            )
        ''')
        conn.commit()
        conn.close()

    def create_profile(self, user_id, name, age, goals, techniques, threshold=0.65):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT OR REPLACE INTO user_profiles VALUES (?, ?, ?, ?, ?, ?)',
                       (user_id, name, age, json.dumps(goals), json.dumps(techniques), threshold))
        conn.commit()
        conn.close()
        logging.info(f"Profile created for {user_id}")

    def get_profile(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
        profile = cursor.fetchone()
        conn.close()
        if profile:
            return {'user_id': profile[0], 'name': profile[1], 'age': profile[2],
                    'goals': json.loads(profile[3]), 'preferred_techniques': json.loads(profile[4]),
                    'anxiety_threshold': profile[5]}
        return None

# CloudStorage
class CloudStorage:
    def __init__(self, config):
        self.config = config
        self.cloud_type = config['cloud_storage']
        if self.cloud_type == 'aws':
            self.s3 = boto3.client('s3', aws_access_key_id=config['aws_access_key'], aws_secret_key=config['aws_secret_key'])
        elif self.cloud_type == 'gcp':
            self.gcs = storage.Client.from_service_account_json(config['gcp_credentials'])

    def upload_file(self, file_path, bucket_name, object_name=None):
        if self.cloud_type == 'aws':
            try:
                self.s3.upload_file(file_path, bucket_name, object_name or os.path.basename(file_path))
                logging.info(f"Uploaded {file_path} to AWS S3")
                return True
            except NoCredentialsError:
                logging.error("AWS credentials not available")
                return False
        elif self.cloud_type == 'gcp':
            try:
                bucket = self.gcs.bucket(bucket_name)
                blob = bucket.blob(object_name or os.path.basename(file_path))
                blob.upload_from_filename(file_path)
                logging.info(f"Uploaded {file_path} to GCP")
                return True
            except Exception as e:
                logging.error(f"GCP upload error: {e}")
                return False
        return False

# EncryptionManager
class EncryptionManager:
    def __init__(self, key):
        self.cipher = Fernet(key)

    def encrypt(self, data):
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data):
        return self.cipher.decrypt(encrypted_data.encode()).decode()

# NotificationManager
class NotificationManager:
    def __init__(self, config):
        self.config = config
        self.twilio_client = Client(config['twilio_sid'], config['twilio_token']) if config['twilio_sid'] else None
        if config['firebase_credentials']:
            cred = credentials.Certificate(config['firebase_credentials'])
            firebase_admin.initialize_app(cred)
        self.support_contacts = {}

    def add_support_contact(self, user_id, phone, threshold=0.9, duration=300):
        if user_id not in self.support_contacts:
            self.support_contacts[user_id] = []
        self.support_contacts[user_id].append({'phone': phone, 'threshold': threshold, 'duration': duration})
        logging.info(f"Support contact added for {user_id}: {phone}")

    def send_sms(self, phone, message):
        if self.twilio_client:
            try:
                self.twilio_client.messages.create(body=message, from_=self.config['twilio_phone'], to=phone)
                logging.info(f"SMS sent to {phone}")
            except Exception as e:
                logging.error(f"SMS send error: {e}")

    def send_push_notification(self, token, title, body):
        try:
            message = messaging.Message(
                notification=messaging.Notification(title=title, body=body),
                token=token
            )
            messaging.send(message)
            logging.info(f"Push notification sent to {token}")
        except Exception as e:
            logging.error(f"Push notification error: {e}")

    def notify_support(self, user_id, anxiety_score, duration):
        if user_id in self.support_contacts:
            for contact in self.support_contacts[user_id]:
                if anxiety_score >= contact['threshold'] and duration >= contact['duration']:
                    self.send_sms(contact['phone'], f"{user_id} has been highly anxious (score: {anxiety_score:.2f}) for {duration//60} minutes. Please check in.")

# VoiceAssistant
class VoiceAssistant:
    def __init__(self, config):
        self.config = config
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.translator = Translator()

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self):
        with sr.Microphone() as source:
            audio = self.recognizer.listen(source)
            try:
                text = self.recognizer.recognize_google(audio, language=self.config['language'])
                logging.info(f"User said: {text}")
                return text
            except Exception as e:
                logging.error(f"Voice recognition error: {e}")
                return None

# Gamification
class Gamification:
    def __init__(self):
        self.points = 0
        self.levels = [100, 500, 1000, 5000]
        self.current_level = 0
        self.badges = []

    def add_points(self, points, activity):
        self.points += points
        if activity == 'cbt_session' and 'CBT Starter' not in self.badges:
            self.badges.append('CBT Starter')
        if self.points >= self.levels[self.current_level]:
            self.current_level += 1
            self.badges.append(f"Level {self.current_level} Achiever")
            logging.info(f"Level up! Current level: {self.current_level}, Points: {self.points}, Badges: {self.badges}")

    def get_status(self):
        return {'points': self.points, 'level': self.current_level, 'badges': self.badges}

# SocialFeatures
class SocialFeatures:
    def __init__(self):
        self.friends = []
        self.groups = []
        self.buddies = {}

    def add_friend(self, friend_id):
        self.friends.append(friend_id)
        logging.info(f"Friend added: {friend_id}")

    def create_group(self, group_name):
        self.groups.append({'name': group_name, 'members': []})
        logging.info(f"Group created: {group_name}")

    def share_progress(self, user_id, group_name, progress):
        for group in self.groups:
            if group['name'] == group_name:
                group['members'].append({'user_id': user_id, 'progress': progress})
                logging.info(f"Progress shared with group {group_name}: {progress}")

    def assign_buddy(self, user_id, buddy_id):
        self.buddies[user_id] = buddy_id
        logging.info(f"Buddy assigned: {user_id} -> {buddy_id}")

# Accessibility
class Accessibility:
    def __init__(self):
        self.high_contrast = False
        self.text_to_speech = False
        self.font_size = 12

    def toggle_high_contrast(self):
        self.high_contrast = not self.high_contrast
        logging.info(f"High contrast mode: {'on' if self.high_contrast else 'off'}")

    def toggle_text_to_speech(self):
        self.text_to_speech = not self.text_to_speech
        logging.info(f"Text to speech mode: {'on' if self.text_to_speech else 'off'}")

    def adjust_font_size(self, size):
        self.font_size = size
        logging.info(f"Font size adjusted to: {size}")

# MoodTracker
class MoodTracker:
    def __init__(self, db_path):
        self.db_path = db_path
        self._initialize_mood_db()

    def _initialize_mood_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS moods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                timestamp TIMESTAMP,
                mood TEXT,
                activity TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log_mood(self, user_id, mood, activity):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO moods (user_id, timestamp, mood, activity) VALUES (?, ?, ?, ?)',
                       (user_id, datetime.datetime.now().isoformat(), mood, activity))
        conn.commit()
        conn.close()
        logging.info(f"Mood logged for {user_id}: {mood}, {activity}")

    def get_mood_history(self, user_id):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM moods WHERE user_id = ?', conn, params=(user_id,))
        conn.close()
        return df

# CBTManager
class CBTManager:
    def __init__(self):
        self.modules = {
            'thought_reframing': [
                {"step": "Identify a negative thought", "prompt": "What’s worrying you right now?"},
                {"step": "Challenge it", "prompt": "Is this thought 100% true? What’s the evidence?"},
                {"step": "Reframe it", "prompt": "What’s a more balanced way to see this?"}
            ]
        }

    def start_session(self, module_name, user_input_func, output_func):
        if module_name not in self.modules:
            output_func("Module not found.")
            return
        responses = []
        for step in self.modules[module_name]:
            output_func(step['prompt'])
            response = user_input_func(step['prompt'])
            responses.append(response)
        output_func(f"Session complete! Your reframed thought: {responses[-1]}")
        return responses

# SoftwareBiofeedback
class SoftwareBiofeedback:
    def __init__(self):
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.mouse_listener = mouse.Listener(on_move=self.on_mouse_move)
        self.key_presses = deque(maxlen=100)
        self.mouse_moves = deque(maxlen=100)
        self.classifier = RandomForestClassifier(n_estimators=50)
        self.scaler = StandardScaler()
        self._train_model()
        self.is_running = False

    def _train_model(self):
        X = np.array([[10, 5], [20, 10], [50, 30], [70, 50]])
        y = np.array([0, 0, 1, 1])
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)

    def on_key_press(self, key):
        self.key_presses.append(time.time())

    def on_mouse_move(self, x, y):
        self.mouse_moves.append((x, y, time.time()))

    def start(self):
        if not self.is_running:
            self.keyboard_listener.start()
            self.mouse_listener.start()
            self.is_running = True
            logging.info("Biofeedback listeners started")
        else:
            logging.info("Biofeedback listeners already running")

    def stop(self):
        if self.is_running:
            self.keyboard_listener.stop()
            self.mouse_listener.stop()
            self.is_running = False
            logging.info("Biofeedback listeners stopped")
        else:
            logging.info("Biofeedback listeners already stopped")

    def get_anxiety_score(self):
        if len(self.key_presses) < 10 or len(self.mouse_moves) < 10:
            return 0.0
        typing_speed = len(self.key_presses) / ((self.key_presses[-1] - self.key_presses[0]) or 1)
        mouse_speed = sum(((x2 - x1)**2 + (y2 - y1)**2)**0.5 for (x1, y1, t1), (x2, y2, t2) in zip(list(self.mouse_moves)[:-1], list(self.mouse_moves)[1:])) / ((self.mouse_moves[-1][2] - self.mouse_moves[0][2]) or 1)
        X = self.scaler.transform([[typing_speed * 100, mouse_speed]])
        prob = self.classifier.predict_proba(X)[0][1]
        return prob

# WearableManager
class WearableManager:
    def __init__(self):
        self.connected = False
        self.data = {'heart_rate': 0, 'skin_conductance': 0}

    def connect(self):
        logging.info("Wearable support placeholder - not implemented yet.")
        return False

    def get_data(self):
        return self.data if self.connected else None

# DefaultSuggestionProvider
class DefaultSuggestionProvider:
    def get_suggestion(self, age, combined_score, user_input):
        if user_input and 'stress' in user_input.lower():
            base = "Try a calming technique."
        else:
            base = "Consider a brief relaxation."
        if age is None:
            return f"{base} How about a deep breath?"
        if combined_score < 0.4:
            if age < 18:
                return f"{base} You’re doing great—want to play a calming game?"
            elif age < 40:
                return f"{base} You’re calm—perfect time for a quick stretch."
            else:
                return f"{base} Nice work staying relaxed—how about a short walk?"
        elif combined_score <= 0.7:
            if age < 18:
                return f"{base} Feeling a bit off? Try your favorite song."
            elif age < 40:
                return f"{base} Stress creeping in? Take a 5-minute break."
            else:
                return f"{base} A little tension? Rest for a bit."
        else:
            if age < 18:
                return f"{base} Things feel heavy—text a friend."
            elif age < 40:
                return f"{base} Pressure’s high—take a walk."
            else:
                return f"{base} You might be overwhelmed—try a nap."
        return base

# InteractionAnalysisSystem
class InteractionAnalysisSystem:
    def __init__(self, config_manager=ConfigManager(), suggestion_provider=DefaultSuggestionProvider()):
        self.config = config_manager.config
        self.suggestion_provider = suggestion_provider
        self.user_id = str(uuid.uuid4())

        # Managers
        self.profile_manager = UserProfileManager(self.config['db_path'])
        self.cloud_storage = CloudStorage(self.config)
        self.encryption_manager = EncryptionManager(self.config['encryption_key'])
        self.notification_manager = NotificationManager(self.config)
        self.voice_assistant = VoiceAssistant(self.config) if self.config['voice_assistant'] != 'off' else None
        self.gamification = Gamification() if self.config['gamification'] else None
        self.social_features = SocialFeatures() if self.config['social_features'] else None
        self.accessibility = Accessibility() if self.config['accessibility'] else None
        self.mood_tracker = MoodTracker(self.config['db_path'])
        self.cbt_manager = CBTManager()
        self.biofeedback = SoftwareBiofeedback()
        self.wearable_manager = WearableManager()

        # Model Setup
        self.db_path = self.config['db_path']
        self.models_dir = self.config['models_dir']
        self.face_detector_model = os.path.join(self.models_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
        self.face_detector_config = os.path.join(self.models_dir, 'deploy.prototxt')
        self.landmark_predictor = os.path.join(self.models_dir, 'shape_predictor_68_face_landmarks.dat')

        if not os.path.exists(self.face_detector_config):
            raise FileNotFoundError(f"Face detector config file not found: {self.face_detector_config}")
        if not os.path.exists(self.face_detector_model):
            raise FileNotFoundError(f"Face detector model file not found: {self.face_detector_model}")
        if not os.path.exists(self.landmark_predictor):
            raise FileNotFoundError(f"Landmark predictor file not found: {self.landmark_predictor}")

        self.face_net = cv2.dnn.readNetFromCaffe(self.face_detector_config, self.face_detector_model)
        self.landmark_detector = dlib.shape_predictor(self.landmark_predictor)
        self.face_detector_hog = dlib.get_frontal_face_detector()

        # Session Data
        self.session_data = {
            'session_id': datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
            'start_time': time.time(),
            'total_frames': 0,
            'frames_with_faces': 0,
            'detected_emotions': {},
            'anxiety_events': 0,
            'recognized_persons': set(),
            'last_alert_time': 0,
            'interaction_log': {},
            'audio_events': [],
            'behavior_logs': [],
            'suggestions': {},
            'user_inputs': {},
            'email_consents': {},
            'chat_history': []
        }
        self.emotion_history = deque(maxlen=300)
        self.anxiety_history = deque(maxlen=300)
        self.timestamps = deque(maxlen=300)
        self.current_faces = {}
        self.next_tracking_id = 1
        self.audio = pyaudio.PyAudio()
        self.audio_stream = None
        self.audio_buffer = deque(maxlen=44100)
        self.audio_running = False
        self.audio_enabled = True  # Flag to track if audio is functional
        self.behavior_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.behavior_scaler = StandardScaler()
        self.chatbot_model_name = self.config['chatbot_model']
        self.tokenizer = AutoTokenizer.from_pretrained(self.chatbot_model_name)
        self.chatbot_model = AutoModelForCausalLM.from_pretrained(self.chatbot_model_name)
        self.app = dash.Dash(__name__)

        self._initialize_database()

    def _initialize_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executescript('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                total_frames INTEGER,
                frames_with_faces INTEGER
            );
            CREATE TABLE IF NOT EXISTS face_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TIMESTAMP,
                person_id TEXT,
                confidence REAL,
                tracking_id INTEGER,
                age INTEGER,
                gender TEXT,
                facial_anxiety_score REAL
            );
        ''')
        conn.commit()
        conn.close()

    def setup_profile(self, root):
        name = simpledialog.askstring("Profile Setup", "Enter your name:", parent=root)
        if name is None:
            return
        age = simpledialog.askinteger("Profile Setup", "Enter your age:", parent=root)
        if age is None:
            return
        goals_input = simpledialog.askfloat("Goals", "Target anxiety reduction (%):", parent=root)
        if goals_input is None:
            return
        goals = {'reduce_anxiety': goals_input}
        techniques_input = simpledialog.askstring("Preferences", "Preferred relaxation techniques (comma-separated):", parent=root)
        techniques = techniques_input.split(',') if techniques_input else []
        self.profile_manager.create_profile(self.user_id, name, age, goals, techniques)
        support_phone = simpledialog.askstring("Support", "Trusted contact phone:", parent=root)
        if support_phone:
            self.notification_manager.add_support_contact(self.user_id, support_phone)

    def log_mood(self, root):
        mood = simpledialog.askstring("Mood", "How are you feeling?", parent=root)
        if mood is None:
            return
        activity = simpledialog.askstring("Activity", "What were you doing?", parent=root)
        if activity is None:
            return
        self.mood_tracker.log_mood(self.user_id, mood, activity)
        if self.gamification:
            self.gamification.add_points(10, 'mood_log')

    def start_cbt_session(self, root):
        def user_input(prompt):
            return simpledialog.askstring("CBT", prompt, parent=root)
        def output(text):
            messagebox.showinfo("CBT", text)
        responses = self.cbt_manager.start_session('thought_reframing', user_input, output)
        if self.gamification and responses:
            self.gamification.add_points(20, 'cbt_session')
        return responses

    def show_mood_history(self, root):
        history = self.mood_tracker.get_mood_history(self.user_id)
        if history.empty:
            messagebox.showinfo("Mood History", "No mood history available.", parent=root)
        else:
            history_str = "\n".join([f"{row['timestamp']}: {row['mood']} ({row['activity']})" for _, row in history.iterrows()])
            messagebox.showinfo("Mood History", history_str, parent=root)

    def show_gamification_status(self, root):
        if not self.gamification:
            messagebox.showinfo("Gamification", "Gamification is disabled.", parent=root)
            return
        status = self.gamification.get_status()
        status_str = f"Points: {status['points']}\nLevel: {status['level']}\nBadges: {', '.join(status['badges']) or 'None'}"
        messagebox.showinfo("Gamification Status", status_str, parent=root)

    def open_settings(self, root):
        settings_window = tk.Toplevel(root)
        settings_window.title("Settings")
        settings_window.geometry("300x200")

        tk.Label(settings_window, text="Accessibility Settings").pack(pady=5)

        high_contrast_var = tk.BooleanVar(value=self.accessibility.high_contrast)
        tk.Checkbutton(settings_window, text="High Contrast", variable=high_contrast_var,
                       command=lambda: self.accessibility.toggle_high_contrast()).pack()

        text_to_speech_var = tk.BooleanVar(value=self.accessibility.text_to_speech)
        tk.Checkbutton(settings_window, text="Text to Speech", variable=text_to_speech_var,
                       command=lambda: self.accessibility.toggle_text_to_speech()).pack()

        tk.Label(settings_window, text="Font Size").pack()
        font_size_var = tk.IntVar(value=self.accessibility.font_size)
        tk.Scale(settings_window, from_=10, to=20, orient=tk.HORIZONTAL, variable=font_size_var,
                 command=lambda val: self.accessibility.adjust_font_size(int(val))).pack()

        tk.Button(settings_window, text="Close", command=settings_window.destroy).pack(pady=10)

    def start_audio_recording(self):
        CHUNK = 1024
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 44100

        # Check for available input devices
        device_count = self.audio.get_device_count()
        input_device_index = None
        for i in range(device_count):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                input_device_index = i
                logging.info(f"Found input device: {device_info['name']}")
                break

        if input_device_index is None:
            logging.error("No audio input devices found. Disabling audio recording.")
            self.audio_enabled = False
            return

        try:
            self.audio_stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=CHUNK
            )
            logging.info("Audio recording started successfully.")
        except Exception as e:
            logging.error(f"Failed to open audio stream: {e}")
            self.audio_enabled = False
            return

        while self.audio_running and self.audio_enabled:
            try:
                data = self.audio_stream.read(CHUNK, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.float32)
                self.audio_buffer.extend(samples)
                mfcc = librosa.feature.mfcc(y=samples, sr=RATE, n_mfcc=13)
                if np.max(mfcc) > 0.5:
                    self.session_data['audio_events'].append({'timestamp': time.time(), 'intensity': np.max(mfcc)})
            except Exception as e:
                logging.error(f"Audio recording error: {e}")
                self.audio_enabled = False
                break

        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                logging.info("Audio stream closed.")
            except Exception as e:
                logging.error(f"Error closing audio stream: {e}")

    def track_faces(self, faces, timestamp):
        new_faces = {}
        for i, (x, y, w, h) in enumerate(faces):
            tid = self.next_tracking_id + i
            new_faces[tid] = {'bbox': (x, y, w, h), 'last_seen': timestamp}
        self.current_faces = new_faces
        self.next_tracking_id += len(faces)
        return list(new_faces.keys())

    def generate_anxiety_report(self, session_id=None):
        session_id = session_id or self.session_data['session_id']
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM face_detections WHERE session_id = ?', conn, params=(session_id,))
        conn.close()
        if not df.empty:
            fig = px.line(df, x='timestamp', y='facial_anxiety_score', title=f"Anxiety Trend - Session {session_id}")
            fig.write_html(f"anxiety_report_{session_id}.html")
            self.cloud_storage.upload_file(f"anxiety_report_{session_id}.html", 'anxiety-reports')
        return df.to_dict()

    def start_api(self):
        from flask import Flask, request, jsonify
        app = Flask(__name__)

        @app.route('/api/v1/anxiety', methods=['GET'])
        def get_anxiety_data():
            session_id = request.args.get('session_id')
            if not session_id:
                return jsonify({'error': 'session_id is required'}), 400
            data = self.generate_anxiety_report(session_id)
            return jsonify(data)

        threading.Thread(target=lambda: app.run(port=self.config['api_port']), daemon=True).start()
        logging.info(f"API running at http://127.0.0.1:{self.config['api_port']}")

    def start_real_time_analysis(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Webcam not opened")
            return
        self.audio_running = True
        threading.Thread(target=self.start_audio_recording, name="AudioRecordingThread").start()
        self.biofeedback.start()

        # Enhanced GUI
        root = tk.Tk()
        root.title("Anxiety Management System")
        root.geometry("400x600")

        # Main Frame
        main_frame = tk.Frame(root, padx=10, pady=10)
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Status Label
        self.status_label = tk.Label(main_frame, text="Anxiety Score: 0.00\nSuggestion: None", font=("Arial", 12), wraplength=350, justify="left")
        self.status_label.pack(pady=10)

        # Buttons Frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="Setup Profile", command=lambda: self.setup_profile(root)).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(button_frame, text="Log Mood", command=lambda: self.log_mood(root)).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(button_frame, text="Start CBT", command=lambda: self.start_cbt_session(root)).grid(row=1, column=0, padx=5, pady=5)
        tk.Button(button_frame, text="Mood History", command=lambda: self.show_mood_history(root)).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(button_frame, text="Gamification", command=lambda: self.show_gamification_status(root)).grid(row=2, column=0, padx=5, pady=5)
        tk.Button(button_frame, text="Settings", command=lambda: self.open_settings(root)).grid(row=2, column=1, padx=5, pady=5)
        tk.Button(button_frame, text="Quit", command=root.quit).grid(row=3, column=0, columnspan=2, pady=10)

        frame_count = 0
        prev_time = 0
        anxiety_duration = 0

        while self.audio_running:
            ret, frame = cap.read()
            if not ret:
                break
            self.session_data['total_frames'] += 1
            frame_count += 1

            if frame_count % self.config['frame_skip'] != 0:
                cv2.imshow('Anxiety Analysis', frame)
                cv2.waitKey(1)
                root.update()
                continue

            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.face_net.setInput(blob)
            detections = self.face_net.forward()

            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.config['confidence_threshold']:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, w_box, h_box) = box.astype("int")
                    faces.append((x, y, w_box, h_box))
                    self.session_data['frames_with_faces'] += 1

            tracking_ids = self.track_faces(faces, time.time())
            bio_anxiety = self.biofeedback.get_anxiety_score()
            combined_score = bio_anxiety

            profile = self.profile_manager.get_profile(self.user_id)
            suggestion = "None"
            if profile:
                suggestion = self.suggestion_provider.get_suggestion(profile['age'], combined_score, self.session_data['user_inputs'].get(1, ''))
                self.session_data['suggestions'][time.time()] = suggestion
                if combined_score > profile['anxiety_threshold']:
                    anxiety_duration += self.config['frame_skip'] / 30
                    self.notification_manager.notify_support(self.user_id, combined_score, anxiety_duration)
                    if self.voice_assistant and time.time() - self.session_data['last_alert_time'] > self.config['alert_cooldown']:
                        self.voice_assistant.speak(suggestion)
                        self.session_data['last_alert_time'] = time.time()
                else:
                    anxiety_duration = 0

            # Update GUI Status
            self.status_label.config(text=f"Anxiety Score: {combined_score:.2f}\nSuggestion: {suggestion}")

            for tid, (x, y, w_box, h_box) in zip(tracking_ids, faces):
                cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
                cv2.putText(frame, f"Anxiety: {combined_score:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time else 0
            prev_time = current_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Anxiety Analysis', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            root.update()

        self.audio_running = False
        cap.release()
        cv2.destroyAllWindows()
        self.biofeedback.stop()
        self.audio.terminate()
        root.destroy()
        self.session_data['end_time'] = time.time()
        conn = sqlite3.connect(self.db_path)
        conn.execute('INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)',
                     (self.session_data['session_id'], self.user_id, self.session_data['start_time'],
                      self.session_data['end_time'], self.session_data['total_frames'], self.session_data['frames_with_faces']))
        conn.commit()
        conn.close()

if __name__ == "__main__":
    system = InteractionAnalysisSystem()
    system.start_real_time_analysis()