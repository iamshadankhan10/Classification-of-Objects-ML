"""
Advanced Measurement Object Detection System
Real-time object detection with distance, height, and speed measurements
"""

import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
from utils.measurement_utils import MeasurementSystem
from utils.camera_utils import initialize_webcam, get_frame_webcam
from config import *

class AdvancedMeasurementDetector:
    """Advanced detector with comprehensive measurement capabilities"""
    
    def __init__(self):
        self.model = None
        self.camera = None
        self.measurement_system = MeasurementSystem()
        self.running = False
        self.frame_count = 0
        self.start_time = time.time()
        
        # Audio settings
        self.audio_enabled = True
        self.last_audio_time = 0
        self.audio_cooldown = 2.0  # seconds
        
    def load_model(self, model_path):
        """Load YOLO model"""
        try:
            self.model = YOLO(model_path)
            print(f"✅ Model loaded successfully: {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
            
    def initialize_camera(self):
        """Initialize camera"""
        try:
            self.camera = initialize_webcam()
            if self.camera is None:
                print("❌ Failed to initialize camera")
                return False
            print("✅ Camera initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
            return False
            
    def speak_measurement(self, text):
        """Speak measurement using text-to-speech"""
        if not self.audio_enabled:
            return
            
        current_time = time.time()
        if current_time - self.last_audio_time < self.audio_cooldown:
            return
            
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.8)
            
            def speak():
                engine.say(text)
                engine.runAndWait()
                
            # Run in separate thread to avoid blocking
            thread = threading.Thread(target=speak)
            thread.daemon = True
            thread.start()
            
            self.last_audio_time = current_time
        except Exception as e:
            print(f"⚠️ TTS Error: {e}")
            
    def process_detections(self, results, frame_time):
        """Process YOLO detections and extract measurement data"""
        detections = []
        
        if results and len(results) > 0:
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        if conf >= CONFIDENCE_THRESHOLD:
                            x1, y1, x2, y2 = box
                            class_name = self.model.names[class_id]
                            
                            detection = {
                                'bbox': [x1, y1, x2, y2],
                                'class': class_name,
                                'confidence': conf,
                                'class_id': class_id
                            }
                            detections.append(detection)
                            
        return detections
        
    def run_detection_loop(self):
        """Main detection loop with measurements"""
        print("🎥 Starting advanced measurement detection...")
        print("📏 Measuring: Distance, Height, Speed")
        print("🔊 Audio announcements enabled")
        print("Press 'q' to quit, 's' for summary, 'a' to toggle audio")
        
        while self.running:
            try:
                # Get frame
                ret, frame = get_frame_webcam(self.camera)
                if not ret:
                    print("❌ Failed to get frame")
                    break
                    
                frame_time = time.time()
                self.frame_count += 1
                
                # Run YOLO detection
                results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
                
                # Process detections
                detections = self.process_detections(results, frame_time)
                
                # Update measurements
                measurements = self.measurement_system.update_tracking(detections, frame_time)
                
                # Draw measurements on frame
                frame = self.measurement_system.draw_measurements(frame, detections, measurements)
                
                # Add frame info
                fps = self.frame_count / (frame_time - self.start_time) if frame_time > self.start_time else 0
                info_text = f"FPS: {fps:.1f} | Objects: {len(detections)} | Frame: {self.frame_count}"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Audio announcements for new measurements
                for i, (detection, measurement) in enumerate(zip(detections, measurements)):
                    if i in measurement and measurement[i]:
                        m = measurement[i]
                        if m['distance'] is not None and m['distance'] < 5.0:  # Only announce close objects
                            audio_text = f"{detection['class']} at {m['distance']:.1f} meters"
                            if m['height'] is not None:
                                audio_text += f", height {m['height']:.1f} meters"
                            if m['speed'] > 0.5:  # Only announce if moving
                                audio_text += f", moving at {m['speed']:.1f} meters per second"
                            
                            self.speak_measurement(audio_text)
                            break  # Only announce one object per frame
                
                # Display frame
                cv2.imshow('Advanced Measurement Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.show_summary(measurements)
                elif key == ord('a'):
                    self.audio_enabled = not self.audio_enabled
                    status = "enabled" if self.audio_enabled else "disabled"
                    print(f"🔊 Audio {status}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error in detection loop: {e}")
                break
                
    def show_summary(self, measurements):
        """Show measurement summary"""
        print("\n" + "="*60)
        print("📊 MEASUREMENT SUMMARY")
        print("="*60)
        
        if not measurements:
            print("No measurements available")
            return
            
        for i, measurement in measurements.items():
            if measurement and i in measurement:
                m = measurement[i]
                print(f"\n🔍 Object {i}:")
                if m['distance'] is not None:
                    print(f"   📏 Distance: {m['distance']:.2f} meters")
                if m['height'] is not None:
                    print(f"   📐 Height: {m['height']:.2f} meters")
                if m['speed'] > 0:
                    print(f"   🏃 Speed: {m['speed']:.2f} m/s")
                if m['object_id'] is not None:
                    print(f"   🆔 Tracking ID: {m['object_id']}")
                    
        print("="*60)
        
    def run(self):
        """Run the advanced measurement detector"""
        print("🚀 ADVANCED MEASUREMENT OBJECT DETECTION SYSTEM")
        print("="*60)
        print("📏 Features: Distance | Height | Speed | Audio")
        print("="*60)
        
        # Load model
        if not self.load_model(MODEL_PATH):
            return False
            
        # Initialize camera
        if not self.initialize_camera():
            return False
            
        # Start detection
        self.running = True
        try:
            self.run_detection_loop()
        finally:
            self.cleanup()
            
        return True
        
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("🧹 Cleanup completed")

def main():
    """Main function"""
    detector = AdvancedMeasurementDetector()
    detector.run()

if __name__ == "__main__":
    main()


