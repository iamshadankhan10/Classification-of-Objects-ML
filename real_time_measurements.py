"""
Real-Time Measurement Display System
Live display of distance, height, and speed measurements
"""

import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
from utils.measurement_utils import MeasurementSystem
from utils.camera_utils import initialize_webcam, get_frame_webcam
from config import *

class RealTimeMeasurementDisplay:
    """Real-time measurement display with comprehensive UI"""
    
    def __init__(self):
        self.model = None
        self.camera = None
        self.measurement_system = MeasurementSystem()
        self.running = False
        self.frame_count = 0
        self.start_time = time.time()
        
        # Display settings
        self.show_measurements = True
        self.show_trajectories = True
        self.show_statistics = True
        
        # Measurement history for statistics
        self.measurement_history = []
        self.max_history = 100
        
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
        
    def draw_enhanced_measurements(self, frame, detections, measurements):
        """Draw enhanced measurement display"""
        for i, (detection, measurement) in enumerate(zip(detections, measurements)):
            if i not in measurement or not measurement[i]:
                continue
                
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            m = measurement[i]
            
            # Draw bounding box with color coding based on distance
            color = self.get_distance_color(m.get('distance', 10))
            thickness = 3 if m.get('distance', 10) < 2 else 2
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # Draw measurement panel
            self.draw_measurement_panel(frame, x1, y1, detection, m)
            
            # Draw trajectory if available
            if self.show_trajectories and m.get('object_id') is not None:
                self.draw_trajectory(frame, m['object_id'])
                
    def get_distance_color(self, distance):
        """Get color based on distance"""
        if distance is None:
            return (0, 255, 255)  # Yellow for unknown
        elif distance < 1.0:
            return (0, 0, 255)    # Red for very close
        elif distance < 2.0:
            return (0, 165, 255)  # Orange for close
        elif distance < 5.0:
            return (0, 255, 0)    # Green for medium
        else:
            return (255, 0, 0)    # Blue for far
            
    def draw_measurement_panel(self, frame, x, y, detection, measurement):
        """Draw detailed measurement panel"""
        panel_width = 200
        panel_height = 120
        panel_x = max(0, int(x))
        panel_y = max(0, int(y) - panel_height - 10)
        
        # Ensure panel fits in frame
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
            f"Class: {detection['class']}",
            f"Conf: {detection['confidence']:.2f}",
        ]
        
        if measurement.get('distance') is not None:
            lines.append(f"Dist: {measurement['distance']:.2f}m")
        if measurement.get('height') is not None:
            lines.append(f"Height: {measurement['height']:.2f}m")
        if measurement.get('speed', 0) > 0:
            lines.append(f"Speed: {measurement['speed']:.2f}m/s")
        if measurement.get('object_id') is not None:
            lines.append(f"ID: {measurement['object_id']}")
            
        # Draw text
        for i, line in enumerate(lines):
            y_pos = panel_y + 20 + i * 20
            cv2.putText(frame, line, (panel_x + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
    def draw_trajectory(self, frame, object_id):
        """Draw object trajectory"""
        if object_id not in self.measurement_system.object_history:
            return
            
        history = self.measurement_system.object_history[object_id]
        if len(history) < 2:
            return
            
        # Draw trajectory line
        points = [(int(pos['centroid'][0]), int(pos['centroid'][1])) for pos in history]
        for i in range(1, len(points)):
            alpha = i / len(points)  # Fade effect
            color = (int(255 * alpha), int(255 * (1 - alpha)), 0)
            cv2.line(frame, points[i-1], points[i], color, 2)
            
        # Draw current position
        if points:
            cv2.circle(frame, points[-1], 5, (0, 255, 0), -1)
            
    def draw_statistics_panel(self, frame):
        """Draw statistics panel"""
        if not self.show_statistics:
            return
            
        # Calculate statistics
        if self.measurement_history:
            distances = [m.get('distance', 0) for m in self.measurement_history if m.get('distance') is not None]
            heights = [m.get('height', 0) for m in self.measurement_history if m.get('height') is not None]
            speeds = [m.get('speed', 0) for m in self.measurement_history if m.get('speed', 0) > 0]
            
            # Statistics text
            stats_lines = [
                "=== MEASUREMENT STATISTICS ===",
                f"Total Objects: {len(self.measurement_history)}",
            ]
            
            if distances:
                stats_lines.extend([
                    f"Avg Distance: {np.mean(distances):.2f}m",
                    f"Min Distance: {np.min(distances):.2f}m",
                    f"Max Distance: {np.max(distances):.2f}m"
                ])
                
            if heights:
                stats_lines.extend([
                    f"Avg Height: {np.mean(heights):.2f}m",
                    f"Min Height: {np.min(heights):.2f}m",
                    f"Max Height: {np.max(heights):.2f}m"
                ])
                
            if speeds:
                stats_lines.extend([
                    f"Avg Speed: {np.mean(speeds):.2f}m/s",
                    f"Max Speed: {np.max(speeds):.2f}m/s"
                ])
                
            # Draw statistics panel
            panel_x, panel_y = 10, frame.shape[0] - 200
            panel_width = 300
            panel_height = len(stats_lines) * 20 + 20
            
            # Background
            cv2.rectangle(frame, (panel_x, panel_y), 
                          (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (panel_x, panel_y), 
                          (panel_x + panel_width, panel_y + panel_height), (0, 255, 0), 2)
            
            # Text
            for i, line in enumerate(stats_lines):
                y_pos = panel_y + 20 + i * 20
                cv2.putText(frame, line, (panel_x + 10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                           
    def update_measurement_history(self, measurements):
        """Update measurement history for statistics"""
        for measurement in measurements.values():
            if measurement:
                self.measurement_history.append(measurement)
                
        # Keep only recent measurements
        if len(self.measurement_history) > self.max_history:
            self.measurement_history = self.measurement_history[-self.max_history:]
            
    def run_detection_loop(self):
        """Main detection loop"""
        print("🎥 Starting real-time measurement display...")
        print("📏 Displaying: Distance | Height | Speed | Trajectories")
        print("Controls: 'q'=quit, 'm'=toggle measurements, 't'=toggle trajectories, 's'=toggle stats")
        
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
                
                # Update history
                self.update_measurement_history(measurements)
                
                # Draw measurements
                if self.show_measurements:
                    self.draw_enhanced_measurements(frame, detections, measurements)
                    
                # Draw statistics
                self.draw_statistics_panel(frame)
                
                # Add frame info
                fps = self.frame_count / (frame_time - self.start_time) if frame_time > self.start_time else 0
                elapsed = frame_time - self.start_time
                
                info_lines = [
                    f"FPS: {fps:.1f}",
                    f"Frame: {self.frame_count}",
                    f"Time: {elapsed:.1f}s",
                    f"Objects: {len(detections)}"
                ]
                
                for i, line in enumerate(info_lines):
                    cv2.putText(frame, line, (10, 30 + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Real-Time Measurements', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.show_measurements = not self.show_measurements
                    print(f"📏 Measurements: {'ON' if self.show_measurements else 'OFF'}")
                elif key == ord('t'):
                    self.show_trajectories = not self.show_trajectories
                    print(f"🛤️  Trajectories: {'ON' if self.show_trajectories else 'OFF'}")
                elif key == ord('s'):
                    self.show_statistics = not self.show_statistics
                    print(f"📊 Statistics: {'ON' if self.show_statistics else 'OFF'}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                break
                
    def run(self):
        """Run the real-time measurement display"""
        print("📊 REAL-TIME MEASUREMENT DISPLAY")
        print("="*50)
        print("📏 Features: Distance | Height | Speed | Trajectories")
        print("🎨 Enhanced UI with color coding and statistics")
        print("="*50)
        
        # Initialize components
        if not self.load_model(MODEL_PATH):
            return False
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
    display = RealTimeMeasurementDisplay()
    display.run()

if __name__ == "__main__":
    main()


