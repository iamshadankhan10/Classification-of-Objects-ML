#!/usr/bin/env python3
"""
Create YOLO v5 Specific Flowchart
Professional flowchart for YOLO v5 object detection system
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_yolov5_flowchart():
    """Create a comprehensive YOLO v5 specific flowchart"""
    
    # Create figure with proper dimensions
    fig, ax = plt.subplots(figsize=(14, 16))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Define YOLO v5 specific flowchart boxes (text, x, y, width, height, color)
    boxes = [
        # Data Collection Phase
        ("Dataset Collection\n(COCO Dataset + Custom Data)", 7, 14.5, 5, 0.9, "#E3F2FD"),
        ("Data Annotation\n(YOLO Format Labels)", 7, 13.0, 5, 0.9, "#E8F5E8"),
        
        # Preprocessing Phase
        ("Data Preprocessing\n(Image Resize, Augmentation)", 7, 11.5, 5, 0.9, "#FFF8E1"),
        ("Data Splitting\n(Train/Validation/Test)", 7, 10.0, 5, 0.9, "#FCE4EC"),
        
        # YOLO v5 Training Phase
        ("YOLO v5 Model Setup\n(Architecture Selection)", 7, 8.5, 5, 0.9, "#E1F5FE"),
        ("YOLO v5 Training\n(PyTorch Implementation)", 7, 7.0, 5, 0.9, "#F3E5F5"),
        ("Model Validation\n(Accuracy & mAP Evaluation)", 7, 5.5, 5, 0.9, "#E8F5E8"),
        
        # Optimization Phase
        ("Model Optimization\n(Quantization & Pruning)", 7, 4.0, 5, 0.9, "#FFF3E0"),
        ("Model Conversion\n(ONNX/TensorRT Format)", 7, 2.5, 5, 0.9, "#FFEBEE"),
        
        # Deployment Phase
        ("System Integration\n(Camera + YOLO v5)", 7, 1.0, 5, 0.9, "#E0F2F1")
    ]
    
    # Draw rectangles and text
    for text, x, y, w, h, color in boxes:
        # Create rounded rectangle
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor="#2C3E50",
            linewidth=2
        )
        ax.add_patch(rect)
        
        # Add text
        ax.text(x, y, text, ha="center", va="center", 
                fontsize=11, fontweight="bold", color="#2C3E50")
    
    # Draw arrows between boxes
    for i in range(len(boxes) - 1):
        _, x, y, _, h, _ = boxes[i]
        _, x2, y2, _, h2, _ = boxes[i + 1]
        
        # Calculate arrow position
        start_y = y - h/2 - 0.15
        end_y = y2 + h2/2 + 0.15
        
        # Draw arrow
        ax.annotate('', xy=(x, end_y), xytext=(x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=3, color='#34495E'))
    
    # Add phase labels on the sides
    phases = [
        ("DATA\nCOLLECTION", 1.5, 13.75, "#E3F2FD"),
        ("PREPROCESSING", 1.5, 10.75, "#FFF8E1"),
        ("YOLO v5\nTRAINING", 1.5, 7.75, "#E1F5FE"),
        ("OPTIMIZATION", 1.5, 3.25, "#FFF3E0"),
        ("DEPLOYMENT", 1.5, 1.0, "#E0F2F1")
    ]
    
    for phase, x, y, color in phases:
        # Create phase indicator
        rect = mpatches.FancyBboxPatch(
            (x - 1.2, y - 0.4), 2.4, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="#2C3E50",
            linewidth=1.5
        )
        ax.add_patch(rect)
        
        # Add phase text
        ax.text(x, y, phase, ha="center", va="center", 
                fontsize=9, fontweight="bold", color="#2C3E50")
    
    # Add YOLO v5 specific features
    features = [
        ("Real-time Detection", 12.5, 13.0),
        ("High Accuracy", 12.5, 11.5),
        ("Fast Inference", 12.5, 10.0),
        ("Mobile Optimized", 12.5, 8.5),
        ("Easy Integration", 12.5, 7.0),
        ("Custom Training", 12.5, 5.5),
        ("Model Export", 12.5, 4.0),
        ("Edge Deployment", 12.5, 2.5)
    ]
    
    for feature, x, y in features:
        ax.text(x, y, f"✓ {feature}", ha="left", va="center", 
                fontsize=10, fontweight="bold", color="#27AE60")
    
    # Add title
    ax.text(7, 15.5, "YOLO v5 Object Detection System - Complete Workflow", 
            ha="center", va="center", fontsize=18, fontweight="bold", color="#2C3E50")
    
    # Add subtitle
    ax.text(7, 15.0, "From Dataset Collection to Real-time Deployment", 
            ha="center", va="center", fontsize=14, style='italic', color="#7F8C8D")
    
    # Add YOLO v5 logo area
    logo_rect = mpatches.FancyBboxPatch(
        (11.5, 14.5), 2.5, 1.5,
        boxstyle="round,pad=0.1",
        facecolor="#FF6B6B",
        edgecolor="#E74C3C",
        linewidth=2
    )
    ax.add_patch(logo_rect)
    ax.text(12.75, 15.25, "YOLO v5", ha="center", va="center", 
            fontsize=14, fontweight="bold", color="white")
    ax.text(12.75, 14.9, "You Only Look Once", ha="center", va="center", 
            fontsize=8, color="white")
    
    # Add performance metrics
    metrics_text = """
YOLO v5 Performance:
• Speed: 140 FPS (V100)
• mAP: 56.8% (COCO)
• Model Size: 14MB
• Inference: <1ms
    """
    
    ax.text(12.5, 6.0, metrics_text, ha="left", va="top", 
            fontsize=9, color="#2C3E50",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9FA", edgecolor="#DEE2E6"))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('yolov5_flowchart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print("✅ YOLO v5 specific flowchart generated successfully!")
    print("📁 Saved as: yolov5_flowchart.png")

def create_yolov5_architecture_diagram():
    """Create YOLO v5 architecture diagram"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # YOLO v5 architecture components
    components = [
        ("Input Image\n(640x640x3)", 2, 6.5, 2, 0.8, "#E3F2FD"),
        ("Backbone\n(CSPDarknet53)", 4, 6.5, 2, 0.8, "#E8F5E8"),
        ("Neck\n(FPN + PAN)", 6, 6.5, 2, 0.8, "#FFF8E1"),
        ("Head\n(Detection)", 8, 6.5, 2, 0.8, "#FCE4EC"),
        ("Output\n(BBox + Class)", 10, 6.5, 2, 0.8, "#E1F5FE")
    ]
    
    # Draw components
    for text, x, y, w, h, color in components:
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="#2C3E50",
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", 
                fontsize=10, fontweight="bold", color="#2C3E50")
    
    # Draw arrows
    for i in range(len(components) - 1):
        _, x, y, _, h, _ = components[i]
        _, x2, y2, _, h2, _ = components[i + 1]
        
        start_x = x + 1
        end_x = x2 - 1
        
        ax.annotate('', xy=(end_x, y2), xytext=(start_x, y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#34495E'))
    
    # Add title
    ax.text(6, 7.5, "YOLO v5 Architecture Overview", 
            ha="center", va="center", fontsize=16, fontweight="bold", color="#2C3E50")
    
    # Add feature descriptions
    features = [
        ("Backbone Features:", 1, 4.5, "left"),
        ("• CSPDarknet53", 1, 4.0, "left"),
        ("• Cross Stage Partial", 1, 3.5, "left"),
        ("• Efficient Feature Extraction", 1, 3.0, "left"),
        ("", 1, 2.5, "left"),
        ("Neck Features:", 6, 4.5, "left"),
        ("• Feature Pyramid Network", 6, 4.0, "left"),
        ("• Path Aggregation Network", 6, 3.5, "left"),
        ("• Multi-scale Feature Fusion", 6, 3.0, "left"),
        ("", 6, 2.5, "left"),
        ("Head Features:", 9, 4.5, "left"),
        ("• Object Detection", 9, 4.0, "left"),
        ("• Classification", 9, 3.5, "left"),
        ("• Bounding Box Regression", 9, 3.0, "left")
    ]
    
    for text, x, y, ha in features:
        ax.text(x, y, text, ha=ha, va="center", 
                fontsize=9, color="#2C3E50")
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('yolov5_architecture.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print("✅ YOLO v5 architecture diagram generated successfully!")
    print("📁 Saved as: yolov5_architecture.png")

def main():
    """Generate YOLO v5 specific visualizations"""
    print("🎨 Creating YOLO v5 specific visualizations...")
    print("=" * 60)
    
    # Create YOLO v5 specific charts
    create_yolov5_flowchart()
    create_yolov5_architecture_diagram()
    
    print("=" * 60)
    print("🎉 YOLO v5 visualizations created successfully!")
    print("📊 Generated files:")
    print("   • yolov5_flowchart.png - Complete YOLO v5 workflow")
    print("   • yolov5_architecture.png - YOLO v5 architecture diagram")

if __name__ == "__main__":
    main()




