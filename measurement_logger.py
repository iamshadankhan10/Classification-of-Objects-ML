"""
Measurement Data Logger
Logs all detection and measurement data to files
"""

import cv2
import numpy as np
import time
import json
import csv
from datetime import datetime
from ultralytics import YOLO
from utils.measurement_utils import MeasurementSystem
from utils.camera_utils import initialize_webcam, get_frame_webcam
from config import *

class MeasurementLogger:
    """Logger for comprehensive measurement data"""
    
    def __init__(self, log_directory="measurement_logs"):
        self.log_directory = log_directory
        self.model = None
        self.camera = None
        self.measurement_system = MeasurementSystem()
        self.running = False
        self.frame_count = 0
        self.start_time = time.time()
        
        # Create log directory
        import os
        os.makedirs(log_directory, exist_ok=True)
        
        # Initialize log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = f"{log_directory}/measurements_{timestamp}.csv"
        self.json_file = f"{log_directory}/measurements_{timestamp}.json"
        self.video_file = f"{log_directory}/detection_video_{timestamp}.mp4"
        
        # Initialize CSV writer
        self.csv_writer = None
        self.csv_file_handle = None
        
        # Initialize video writer
        self.video_writer = None
        
        # Data storage
        self.all_measurements = []
        
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
            
    def initialize_logging(self):
        """Initialize logging files"""
        try:
            # CSV setup
            self.csv_file_handle = open(self.csv_file, 'w', newline='')
            fieldnames = [
                'timestamp', 'frame_number', 'object_id', 'class_name', 'confidence',
                'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                'distance_m', 'height_m', 'speed_ms', 'centroid_x', 'centroid_y'
            ]
            self.csv_writer = csv.DictWriter(self.csv_file_handle, fieldnames=fieldnames)
            self.csv_writer.writeheader()
            
            # Video setup
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.video_file, fourcc, 30.0, (640, 480)
            )
            
            print(f"📝 Logging to: {self.csv_file}")
            print(f"🎥 Video to: {self.video_file}")
            return True
            
        except Exception as e:
            print(f"❌ Logging initialization error: {e}")
            return False
            
    def log_measurements(self, detections, measurements, frame_number, timestamp):
        """Log measurement data"""
        for i, (detection, measurement) in enumerate(zip(detections, measurements)):
            if i in measurement and measurement[i]:
                m = measurement[i]
                
                # Calculate centroid
                bbox = detection['bbox']
                centroid_x = (bbox[0] + bbox[2]) / 2
                centroid_y = (bbox[1] + bbox[3]) / 2
                
                # Prepare row data
                row_data = {
                    'timestamp': timestamp,
                    'frame_number': frame_number,
                    'object_id': m.get('object_id', -1),
                    'class_name': detection['class'],
                    'confidence': detection['confidence'],
                    'bbox_x1': bbox[0],
                    'bbox_y1': bbox[1],
                    'bbox_x2': bbox[2],
                    'bbox_y2': bbox[3],
                    'distance_m': m.get('distance'),
                    'height_m': m.get('height'),
                    'speed_ms': m.get('speed', 0.0),
                    'centroid_x': centroid_x,
                    'centroid_y': centroid_y
                }
                
                # Write to CSV
                self.csv_writer.writerow(row_data)
                
                # Store for JSON
                self.all_measurements.append(row_data)
                
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
        
    def run_logging_session(self, duration_seconds=60):
        """Run measurement logging session"""
        print(f"📊 Starting measurement logging for {duration_seconds} seconds...")
        print("📏 Logging: Distance, Height, Speed, Position")
        print("🎥 Recording video with measurements")
        print("Press 'q' to stop early")
        
        start_time = time.time()
        
        while self.running and (time.time() - start_time) < duration_seconds:
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
                
                # Log measurements
                timestamp = datetime.now().isoformat()
                self.log_measurements(detections, measurements, self.frame_count, timestamp)
                
                # Draw measurements on frame
                frame = self.measurement_system.draw_measurements(frame, detections, measurements)
                
                # Add logging info
                elapsed = time.time() - start_time
                remaining = duration_seconds - elapsed
                info_text = f"Logging: {elapsed:.1f}s | Remaining: {remaining:.1f}s | Objects: {len(detections)}"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Write video frame
                self.video_writer.write(frame)
                
                # Display frame
                cv2.imshow('Measurement Logger', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                break
                
        print(f"✅ Logging completed. Processed {self.frame_count} frames")
        
    def save_json_summary(self):
        """Save comprehensive JSON summary"""
        summary = {
            'session_info': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'duration_seconds': time.time() - self.start_time,
                'total_frames': self.frame_count,
                'fps': self.frame_count / (time.time() - self.start_time) if time.time() > self.start_time else 0
            },
            'measurements': self.all_measurements,
            'statistics': self.calculate_statistics()
        }
        
        with open(self.json_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"📄 JSON summary saved: {self.json_file}")
        
    def calculate_statistics(self):
        """Calculate measurement statistics"""
        if not self.all_measurements:
            return {}
            
        stats = {
            'total_detections': len(self.all_measurements),
            'unique_objects': len(set(m['object_id'] for m in self.all_measurements if m['object_id'] != -1)),
            'class_distribution': {},
            'distance_stats': {},
            'height_stats': {},
            'speed_stats': {}
        }
        
        # Class distribution
        classes = [m['class_name'] for m in self.all_measurements]
        for cls in set(classes):
            stats['class_distribution'][cls] = classes.count(cls)
            
        # Distance statistics
        distances = [m['distance_m'] for m in self.all_measurements if m['distance_m'] is not None]
        if distances:
            stats['distance_stats'] = {
                'min': min(distances),
                'max': max(distances),
                'avg': sum(distances) / len(distances),
                'count': len(distances)
            }
            
        # Height statistics
        heights = [m['height_m'] for m in self.all_measurements if m['height_m'] is not None]
        if heights:
            stats['height_stats'] = {
                'min': min(heights),
                'max': max(heights),
                'avg': sum(heights) / len(heights),
                'count': len(heights)
            }
            
        # Speed statistics
        speeds = [m['speed_ms'] for m in self.all_measurements if m['speed_ms'] > 0]
        if speeds:
            stats['speed_stats'] = {
                'min': min(speeds),
                'max': max(speeds),
                'avg': sum(speeds) / len(speeds),
                'count': len(speeds)
            }
            
        return stats
        
    def run(self, duration_seconds=60):
        """Run measurement logger"""
        print("📊 MEASUREMENT DATA LOGGER")
        print("="*50)
        print("📏 Logging: Distance | Height | Speed | Position")
        print("📁 Output: CSV | JSON | Video")
        print("="*50)
        
        # Initialize components
        if not self.load_model(MODEL_PATH):
            return False
        if not self.initialize_camera():
            return False
        if not self.initialize_logging():
            return False
            
        # Start logging
        self.running = True
        try:
            self.run_logging_session(duration_seconds)
        finally:
            self.cleanup()
            
        return True
        
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        
        # Close files
        if self.csv_file_handle:
            self.csv_file_handle.close()
        if self.video_writer:
            self.video_writer.release()
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        
        # Save JSON summary
        self.save_json_summary()
        
        print("🧹 Cleanup completed")
        print(f"📁 Files saved in: {self.log_directory}/")

def main():
    """Main function"""
    logger = MeasurementLogger()
    duration = 60  # seconds
    print(f"⏱️  Logging for {duration} seconds...")
    logger.run(duration)

if __name__ == "__main__":
    main()


