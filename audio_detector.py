#!/usr/bin/env python3
"""
Audio-Enhanced Object Detection System
Detects objects and announces them through audio feedback
"""

import cv2
import sys
import os
import time
import threading
from datetime import datetime
from ultralytics import YOLO
from config import *
from utils.camera_utils import initialize_webcam, initialize_realsense, get_frame_webcam, get_frame_realsense, test_camera
from utils.detection_utils import draw_detections, get_detection_summary, draw_summary_text

# Try to import text-to-speech libraries
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️ pyttsx3 not available. Install with: pip install pyttsx3")

try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("⚠️ gTTS not available. Install with: pip install gtts pygame")

class AudioDetector:
    def __init__(self):
        self.detection_log = []
        self.frame_count = 0
        self.start_time = time.time()
        self.detected_objects = set()
        self.object_counts = {}
        self.last_announcement = {}
        self.announcement_cooldown = 3.0  # seconds between same object announcements
        
        # Initialize TTS
        self.tts_engine = None
        self.audio_enabled = True
        
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.setup_tts_voice()
                print("✅ Audio system initialized with pyttsx3")
            except Exception as e:
                print(f"⚠️ Failed to initialize pyttsx3: {e}")
                self.audio_enabled = False
        elif GTTS_AVAILABLE:
            print("✅ Audio system initialized with gTTS")
        else:
            print("❌ No TTS library available. Audio disabled.")
            self.audio_enabled = False
    
    def setup_tts_voice(self):
        """Setup TTS voice properties"""
        if self.tts_engine:
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to find a female voice
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                else:
                    # Use first available voice
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            # Set speech rate and volume
            self.tts_engine.setProperty('rate', 150)  # Speed of speech
            self.tts_engine.setProperty('volume', 0.8)  # Volume level
    
    def announce_detection(self, object_name, confidence):
        """Announce detected object through audio"""
        if not self.audio_enabled:
            return
        
        # Check cooldown to avoid spam
        current_time = time.time()
        if object_name in self.last_announcement:
            if current_time - self.last_announcement[object_name] < self.announcement_cooldown:
                return
        
        self.last_announcement[object_name] = current_time
        
        # Create announcement text
        confidence_percent = int(confidence * 100)
        announcement = f"Detected {object_name} with {confidence_percent} percent confidence"
        
        print(f"🔊 Audio: {announcement}")
        
        # Use threading to avoid blocking detection
        if TTS_AVAILABLE and self.tts_engine:
            threading.Thread(target=self._speak_pyttsx3, args=(announcement,), daemon=True).start()
        elif GTTS_AVAILABLE:
            threading.Thread(target=self._speak_gtts, args=(announcement,), daemon=True).start()
    
    def _speak_pyttsx3(self, text):
        """Speak using pyttsx3"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"⚠️ TTS Error: {e}")
    
    def _speak_gtts(self, text):
        """Speak using gTTS"""
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save("temp_audio.mp3")
            
            pygame.mixer.init()
            pygame.mixer.music.load("temp_audio.mp3")
            pygame.mixer.music.play()
            
            # Wait for audio to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Clean up
            if os.path.exists("temp_audio.mp3"):
                os.remove("temp_audio.mp3")
                
        except Exception as e:
            print(f"⚠️ gTTS Error: {e}")
    
    def announce_summary(self, summary):
        """Announce detection summary"""
        if not self.audio_enabled:
            return
        
        total_objects = summary['total_objects']
        if total_objects > 0:
            announcement = f"Total {total_objects} objects detected"
            print(f"🔊 Summary: {announcement}")
            
            if TTS_AVAILABLE and self.tts_engine:
                threading.Thread(target=self._speak_pyttsx3, args=(announcement,), daemon=True).start()
            elif GTTS_AVAILABLE:
                threading.Thread(target=self._speak_gtts, args=(announcement,), daemon=True).start()
    
    def log_detection(self, frame, results, summary):
        """Log detection with audio announcements"""
        timestamp = datetime.now().isoformat()
        frame_data = {
            'timestamp': timestamp,
            'frame_number': self.frame_count,
            'total_objects': summary['total_objects'],
            'average_confidence': summary['average_confidence'],
            'detections': []
        }
        
        # Process YOLO detections
        if results and len(results) > 0 and results[0].boxes:
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = results[0].names[class_id] if hasattr(results[0], 'names') else f"class_{class_id}"
                
                # Track objects
                self.detected_objects.add(class_name)
                self.object_counts[class_name] = self.object_counts.get(class_name, 0) + 1
                
                # Announce detection
                self.announce_detection(class_name, confidence)
                
                detection = {
                    'object_id': i,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bounding_box': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    },
                    'center': {
                        'x': float((x1 + x2) / 2),
                        'y': float((y1 + y2) / 2)
                    }
                }
                frame_data['detections'].append(detection)
        
        self.detection_log.append(frame_data)
        self.frame_count += 1
        
        # Print real-time detection info
        if summary['total_objects'] > 0:
            objects = [det['class_name'] for det in frame_data['detections']]
            print(f"🔍 Frame {self.frame_count}: {', '.join(objects)}")
        
        return frame_data
    
    def print_detection_summary(self):
        """Print comprehensive detection summary"""
        print("\n" + "="*60)
        print("📊 AUDIO-ENHANCED DETECTION SUMMARY")
        print("="*60)
        
        total_detections = sum(len(frame['detections']) for frame in self.detection_log)
        unique_objects = len(self.detected_objects)
        
        print(f"📈 Session Duration: {time.time() - self.start_time:.1f} seconds")
        print(f"📈 Total Frames Processed: {self.frame_count}")
        print(f"📈 Total Objects Detected: {total_detections}")
        print(f"📈 Unique Object Types: {unique_objects}")
        print(f"🔊 Audio System: {'Enabled' if self.audio_enabled else 'Disabled'}")
        
        print(f"\n🎯 ALL OBJECTS DETECTED:")
        print("-" * 40)
        for obj, count in sorted(self.object_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {obj}: {count} detections")
        
        print(f"\n🔍 DETECTED OBJECT TYPES:")
        print("-" * 40)
        for obj in sorted(self.detected_objects):
            print(f"  ✓ {obj}")
        
        return {
            'total_detections': total_detections,
            'unique_objects': unique_objects,
            'object_counts': self.object_counts,
            'detected_objects': list(self.detected_objects),
            'audio_enabled': self.audio_enabled
        }

def load_model(model_path: str):
    """Load YOLO model with error handling"""
    try:
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None
        
        model = YOLO(model_path)
        print(f"Model loaded successfully: {model_path}")
        print(f"Available classes: {model.names}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def initialize_camera():
    """Initialize camera (webcam or RealSense)"""
    try:
        realsense_pipeline = initialize_realsense()
        if realsense_pipeline:
            print("RealSense camera initialized successfully")
            return realsense_pipeline, get_frame_realsense
    except Exception as e:
        print(f"RealSense camera not available: {e}")
    
    print("Using webcam...")
    if not test_camera(CAMERA_INDEX):
        print(f"Camera {CAMERA_INDEX} not available. Trying camera 1...")
        if not test_camera(1):
            print("No cameras available!")
            return None, None
    
    webcam = initialize_webcam(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)
    return webcam, get_frame_webcam

def main():
    """Main application with audio-enhanced detection"""
    print("="*60)
    print("🔊 AUDIO-ENHANCED OBJECT DETECTION SYSTEM")
    print("="*60)
    print("📋 Detecting objects with real-time audio announcements")
    print("📋 Press 'q' to quit, 's' for summary, 'a' to toggle audio")
    print("="*60)
    
    # Load model
    model = load_model(MODEL_PATH)
    if model is None:
        print("Failed to load model. Exiting...")
        return
    
    # Initialize camera
    camera, get_frame_func = initialize_camera()
    if camera is None:
        print("Failed to initialize camera. Exiting...")
        return
    
    # Initialize audio detector
    detector = AudioDetector()
    
    print("\n🎥 Starting audio-enhanced detection...")
    print("Objects will be announced through audio as they are detected!")
    
    try:
        while True:
            # Get frame
            depth_frame, color_frame = get_frame_func(camera)
            
            if color_frame is None:
                print("Failed to read frame")
                continue
            
            # Run detection
            if TRACK_OBJECTS:
                results = model.track(color_frame, classes=DETECT_CLASSES, 
                                   conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE)
            else:
                results = model(color_frame, classes=DETECT_CLASSES, 
                              conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE)
            
            # Get detection summary
            summary = get_detection_summary(results)
            
            # Log detection with audio
            frame_data = detector.log_detection(color_frame, results, summary)
            
            # Draw detections and summary
            annotated_frame = draw_detections(color_frame, results, 
                                           show_confidence=SHOW_CONFIDENCE,
                                           show_class_names=SHOW_CLASS_NAMES)
            annotated_frame = draw_summary_text(annotated_frame, summary, 
                                             TEXT_POSITION, FONT_SCALE, 
                                             FONT_THICKNESS, TEXT_COLOR)
            
            # Display frame
            cv2.imshow(WINDOW_NAME, annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Print current summary
                detector.print_detection_summary()
            elif key == ord('a'):
                # Toggle audio
                detector.audio_enabled = not detector.audio_enabled
                status = "enabled" if detector.audio_enabled else "disabled"
                print(f"🔊 Audio {status}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        if hasattr(camera, 'release'):
            camera.release()
        elif hasattr(camera, 'stop'):
            camera.stop()
        cv2.destroyAllWindows()
        
        # Print final summary
        print("\n" + "="*60)
        print("📊 FINAL AUDIO-ENHANCED DETECTION SUMMARY")
        print("="*60)
        summary = detector.print_detection_summary()
        
        print("Application closed")

if __name__ == "__main__":
    main()



