#!/usr/bin/env python3
"""
Generate Object Detection Flowchart
Creates a professional flowchart for the object detection system
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_flowchart():
    """Create and save the object detection flowchart"""
    
    # Create figure with high DPI for quality
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define flowchart boxes (text, x, y, color)
    boxes = [
        ("Dataset Collection\n(Custom / COCO Dataset)", 5, 10.5, "#E8F4FD"),
        ("Data Preprocessing\n(Resize, Label, Normalize)", 5, 9.2, "#D1E7DD"),
        ("Model Training\n(YOLO v8 / TensorFlow)", 5, 7.9, "#FFF3CD"),
        ("Model Testing & Validation\n(Accuracy Evaluation)", 5, 6.6, "#F8D7DA"),
        ("Model Optimization\n(Lightweight Conversion)", 5, 5.3, "#D4E6F1"),
        ("System Integration\n(Camera Setup, Libraries)", 5, 4.0, "#E8DAEF"),
        ("Real-time Detection\n(Live Object Recognition)", 5, 2.7, "#D5F4E6"),
        ("Performance Monitoring\n(Latency, FPS, Accuracy)", 5, 1.4, "#FCF3CF")
    ]
    
    # Draw rectangles and text
    for text, x, y, color in boxes:
        # Create rounded rectangle
        rect = mpatches.FancyBboxPatch(
            (x - 3.5, y - 0.5), 7.0, 1.0,
            boxstyle="round,pad=0.1",
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
        _, x, y, _ = boxes[i]
        _, x2, y2, _ = boxes[i + 1]
        
        # Calculate arrow position
        start_y = y - 0.6
        end_y = y2 + 0.6
        
        # Draw arrow
        ax.annotate('', xy=(x, end_y), xytext=(x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#34495E'))
    
    # Add title
    ax.text(5, 11.5, "Object Detection System - Development Pipeline", 
            ha="center", va="center", fontsize=16, fontweight="bold", color="#2C3E50")
    
    # Add subtitle
    ax.text(5, 11.1, "From Dataset to Real-time Implementation", 
            ha="center", va="center", fontsize=12, style='italic', color="#7F8C8D")
    
    # Add legend/key
    legend_elements = [
        mpatches.Patch(color='#E8F4FD', label='Data Collection'),
        mpatches.Patch(color='#D1E7DD', label='Preprocessing'),
        mpatches.Patch(color='#FFF3CD', label='Training'),
        mpatches.Patch(color='#F8D7DA', label='Validation'),
        mpatches.Patch(color='#D4E6F1', label='Optimization'),
        mpatches.Patch(color='#E8DAEF', label='Integration'),
        mpatches.Patch(color='#D5F4E6', label='Deployment'),
        mpatches.Patch(color='#FCF3CF', label='Monitoring')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('object_detection_flowchart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print("✅ Flowchart generated successfully!")
    print("📁 Saved as: object_detection_flowchart.png")
    print("📊 High resolution (300 DPI) image ready for documentation")

if __name__ == "__main__":
    create_flowchart()
