"""
Demo Script for Advanced Measurement System
Showcases distance, height, and speed measurement capabilities
"""

import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
from utils.measurement_utils import MeasurementSystem
from utils.camera_utils import initialize_webcam, get_frame_webcam
from config import *

class MeasurementDemo:
    """Demo class showcasing all measurement capabilities"""
    
    def __init__(self):
        self.model = None
        self.camera = None
        self.measurement_system = MeasurementSystem()
        self.running = False
        self.frame_count = 0
        self.start_time = time.time()
        
        # Demo settings
        self.demo_mode = "comprehensive"  # comprehensive, distance_only, speed_only
        self.show_audio = True
        self.audio_cooldown = 3.0
        self.last_audio_time = 0
        
    def load_model(self, model_path):
        """Load YOLO model"""
        try:
            self.model = YOLO(model_path)
            print(f"✅ Model loaded: {model_path}")
            return True
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            return False
            
    def initialize_camera(self):
        """Initialize camera"""
        try:
            self.camera = initialize_webcam()
            if self.camera is None:
                return False
            print("✅ Camera initialized")
            return True
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False
            
    def speak_measurement(self, text):
        """Speak measurement using text-to-speech"""
        if not self.show_audio:
            return
            
        current_time = time.time()
        if current_time - self.last_audio_time < self.audio_cooldown:
            return
            
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 120)
            engine.setProperty('volume', 0.7)
            
            def speak():
                engine.say(text)
                engine.runAndWait()
                
            thread = threading.Thread(target=speak)
            thread.daemon = True
            thread.start()
            
            self.last_audio_time = current_time
        except Exception as e:
            print(f"⚠️ TTS Error: {e}")
            
    def process_detections(self, results, frame_time):
        """Process YOLO detections"""
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
        
    def draw_comprehensive_measurements(self, frame, detections, measurements):
        """Draw comprehensive measurement display"""
        for i, (detection, measurement) in enumerate(zip(detections, measurements)):
            if i not in measurement or not measurement[i]:
                continue
                
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            m = measurement[i]
            
            # Color coding based on distance
            if m.get('distance') is not None:
                if m['distance'] < 1.0:
                    color = (0, 0, 255)  # Red - very close
                elif m['distance'] < 2.0:
                    color = (0, 165, 255)  # Orange - close
                elif m['distance'] < 5.0:
                    color = (0, 255, 0)  # Green - medium
                else:
                    color = (255, 0, 0)  # Blue - far
            else:
                color = (0, 255, 255)  # Yellow - unknown distance
                
            # Draw bounding box
            thickness = 3 if m.get('distance', 10) < 2 else 2
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # Draw measurement info
            self.draw_measurement_info(frame, x1, y1, detection, m)
            
    def draw_measurement_info(self, frame, x, y, detection, measurement):
        """Draw detailed measurement information"""
        # Create info panel
        panel_width = 250
        panel_height = 100
        panel_x = max(0, int(x))
        panel_y = max(0, int(y) - panel_height - 10)
        
        # Ensure panel fits
        if panel_x + panel_width > frame.shape[1]:
            panel_x = frame.shape[1] - panel_width
        if panel_y < 0:
            panel_y = int(y) + 10
            
        # Draw panel background
        cv2.rectangle(frame, (panel_x, panel_y), 
                      (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), 
                      (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)
        
        # Panel content
        lines = [
            f"Object: {detection['class']} ({detection['confidence']:.2f})",
        ]
        
        if measurement.get('distance') is not None:
            lines.append(f"📏 Distance: {measurement['distance']:.2f} meters")
        if measurement.get('height') is not None:
            lines.append(f"📐 Height: {measurement['height']:.2f} meters")
        if measurement.get('speed', 0) > 0.1:
            lines.append(f"🏃 Speed: {measurement['speed']:.2f} m/s")
        if measurement.get('object_id') is not None:
            lines.append(f"🆔 ID: {measurement['object_id']}")
            
        # Draw text
        for i, line in enumerate(lines):
            y_pos = panel_y + 20 + i * 18
            cv2.putText(frame, line, (panel_x + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
    def draw_demo_info(self, frame):
        """Draw demo information panel"""
        info_lines = [
            "🎯 ADVANCED MEASUREMENT DEMO",
            "📏 Distance: Object-to-camera distance",
            "📐 Height: Real-world object height", 
            "🏃 Speed: Object movement speed",
            "🎨 Colors: Red=close, Orange=near, Green=medium, Blue=far",
            "",
            "Controls: 'q'=quit, 'a'=toggle audio, 'd'=distance only",
            "         's'=speed only, 'c'=comprehensive"
        ]
        
        # Draw background
        panel_x, panel_y = 10, 10
        panel_width = 400
        panel_height = len(info_lines) * 20 + 20
        
        cv2.rectangle(frame, (panel_x, panel_y), 
                      (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), 
                      (panel_x + panel_width, panel_y + panel_height), (0, 255, 0), 2)
        
        # Draw text
        for i, line in enumerate(info_lines):
            y_pos = panel_y + 20 + i * 20
            color = (0, 255, 0) if line.startswith("🎯") else (255, 255, 255)
            cv2.putText(frame, line, (panel_x + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                       
    def run_demo_loop(self):
        """Main demo loop"""
        print("🎯 ADVANCED MEASUREMENT DEMO")
        print("="*50)
        print("📏 Measuring: Distance | Height | Speed")
        print("🎨 Color-coded by distance")
        print("🔊 Audio announcements enabled")
        print("="*50)
        print("Controls: 'q'=quit, 'a'=audio, 'd'=distance, 's'=speed, 'c'=comprehensive")
        
        while self.running:
            try:
                # Get frame
                ret, frame = get_frame_webcam(self.camera)
                if not ret:
                    print("❌ Frame capture failed")
                    break
                    
                frame_time = time.time()
                self.frame_count += 1
                
                # Run detection
                results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
                detections = self.process_detections(results, frame_time)
                
                # Update measurements
                measurements = self.measurement_system.update_tracking(detections, frame_time)
                
                # Draw measurements based on demo mode
                if self.demo_mode == "comprehensive":
                    self.draw_comprehensive_measurements(frame, detections, measurements)
                elif self.demo_mode == "distance_only":
                    # Show only distance measurements
                    for i, (detection, measurement) in enumerate(zip(detections, measurements)):
                        if i in measurement and measurement[i] and measurement[i].get('distance'):
                            self.draw_distance_only(frame, detection, measurement[i])
                elif self.demo_mode == "speed_only":
                    # Show only speed measurements
                    for i, (detection, measurement) in enumerate(zip(detections, measurements)):
                        if i in measurement and measurement[i] and measurement[i].get('speed', 0) > 0:
                            self.draw_speed_only(frame, detection, measurement[i])
                
                # Draw demo info
                self.draw_demo_info(frame)
                
                # Add frame info
                fps = self.frame_count / (frame_time - self.start_time) if frame_time > self.start_time else 0
                elapsed = frame_time - self.start_time
                
                status_text = f"Mode: {self.demo_mode} | FPS: {fps:.1f} | Objects: {len(detections)} | Time: {elapsed:.1f}s"
                cv2.putText(frame, status_text, (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Audio announcements
                for i, (detection, measurement) in enumerate(zip(detections, measurements)):
                    if i in measurement and measurement[i]:
                        m = measurement[i]
                        if m.get('distance') is not None and m['distance'] < 3.0:
                            audio_text = f"{detection['class']} at {m['distance']:.1f} meters"
                            if m.get('height') is not None:
                                audio_text += f", height {m['height']:.1f} meters"
                            if m.get('speed', 0) > 0.5:
                                audio_text += f", moving at {m['speed']:.1f} meters per second"
                            
                            self.speak_measurement(audio_text)
                            break
                
                # Display frame
                cv2.imshow('Advanced Measurement Demo', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    self.show_audio = not self.show_audio
                    print(f"🔊 Audio: {'ON' if self.show_audio else 'OFF'}")
                elif key == ord('d'):
                    self.demo_mode = "distance_only"
                    print("📏 Mode: Distance measurements only")
                elif key == ord('s'):
                    self.demo_mode = "speed_only"
                    print("🏃 Mode: Speed measurements only")
                elif key == ord('c'):
                    self.demo_mode = "comprehensive"
                    print("📊 Mode: Comprehensive measurements")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                break
                
    def draw_distance_only(self, frame, detection, measurement):
        """Draw distance-only measurements"""
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # Color based on distance
        distance = measurement.get('distance', 10)
        if distance < 1.0:
            color = (0, 0, 255)  # Red
        elif distance < 2.0:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green
            
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        
        # Distance text
        text = f"Distance: {distance:.2f}m"
        cv2.putText(frame, text, (int(x1), int(y1) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                   
    def draw_speed_only(self, frame, detection, measurement):
        """Draw speed-only measurements"""
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        speed = measurement.get('speed', 0)
        if speed < 0.5:
            color = (0, 255, 0)  # Green - slow
        elif speed < 1.0:
            color = (0, 165, 255)  # Orange - medium
        else:
            color = (0, 0, 255)  # Red - fast
            
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        
        # Speed text
        text = f"Speed: {speed:.2f} m/s"
        cv2.putText(frame, text, (int(x1), int(y1) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                   
    def run(self):
        """Run the measurement demo"""
        # Initialize components
        if not self.load_model(MODEL_PATH):
            return False
        if not self.initialize_camera():
            return False
            
        # Start demo
        self.running = True
        try:
            self.run_demo_loop()
        finally:
            self.cleanup()
            
        return True
        
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("🧹 Demo completed")

def main():
    """Main function"""
    demo = MeasurementDemo()
    demo.run()

if __name__ == "__main__":
    main()


