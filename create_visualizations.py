#!/usr/bin/env python3
"""
Create Professional Visualizations for Object Detection Project
- Clean flowchart without overlapping
- Pie chart showing technology distribution
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def create_clean_flowchart():
    """Create a clean flowchart without overlapping elements"""
    
    # Create figure with proper spacing
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define flowchart boxes with proper spacing (text, x, y, width, height, color)
    boxes = [
        ("Dataset Collection\n(Custom / COCO Dataset)", 6, 12.5, 4, 0.8, "#E3F2FD"),
        ("Data Preprocessing\n(Resize, Label, Normalize)", 6, 11.2, 4, 0.8, "#E8F5E8"),
        ("Model Training\n(YOLO v5 / PyTorch)", 6, 9.9, 4, 0.8, "#FFF8E1"),
        ("Model Testing & Validation\n(Accuracy Evaluation)", 6, 8.6, 4, 0.8, "#FCE4EC"),
        ("Model Optimization\n(Lightweight Conversion)", 6, 7.3, 4, 0.8, "#E1F5FE"),
        ("System Integration\n(Camera Setup, Libraries)", 6, 6.0, 4, 0.8, "#F3E5F5"),
        ("Real-time Detection\n(Live Object Recognition)", 6, 4.7, 4, 0.8, "#E8F5E8"),
        ("Performance Monitoring\n(Latency, FPS, Accuracy)", 6, 3.4, 4, 0.8, "#FFF3E0")
    ]
    
    # Draw rectangles and text
    for text, x, y, w, h, color in boxes:
        # Create rounded rectangle
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="#2C3E50",
            linewidth=1.5
        )
        ax.add_patch(rect)
        
        # Add text
        ax.text(x, y, text, ha="center", va="center", 
                fontsize=10, fontweight="bold", color="#2C3E50")
    
    # Draw arrows between boxes
    for i in range(len(boxes) - 1):
        _, x, y, _, h, _ = boxes[i]
        _, x2, y2, _, h2, _ = boxes[i + 1]
        
        # Calculate arrow position
        start_y = y - h/2 - 0.1
        end_y = y2 + h2/2 + 0.1
        
        # Draw arrow
        ax.annotate('', xy=(x, end_y), xytext=(x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#34495E'))
    
    # Add title
    ax.text(6, 13.5, "Object Detection System - Development Pipeline", 
            ha="center", va="center", fontsize=16, fontweight="bold", color="#2C3E50")
    
    # Add subtitle
    ax.text(6, 13.1, "Complete workflow from data collection to deployment", 
            ha="center", va="center", fontsize=12, style='italic', color="#7F8C8D")
    
    # Add process indicators
    process_labels = ["Data Phase", "Training Phase", "Optimization Phase", "Deployment Phase"]
    process_x = [1.5, 3.5, 8.5, 10.5]
    process_y = [11.8, 7.95, 5.65, 2.3]
    
    for label, x, y in zip(process_labels, process_x, process_y):
        ax.text(x, y, label, ha="center", va="center", 
                fontsize=9, style='italic', color="#7F8C8D",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9FA", edgecolor="#DEE2E6"))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('clean_flowchart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print("✅ Clean flowchart generated successfully!")
    print("📁 Saved as: clean_flowchart.png")

def create_technology_piechart():
    """Create a pie chart showing technology distribution"""
    
    # Technology categories and their percentages
    technologies = {
        'YOLO v5 (AI/ML)': 25,
        'OpenCV (Computer Vision)': 20,
        'Python (Core Language)': 15,
        'PyTorch (Deep Learning)': 12,
        'NumPy (Data Processing)': 8,
        'Matplotlib (Visualization)': 5,
        'Intel RealSense (Hardware)': 5,
        'Custom CV Algorithms': 10
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors for each technology
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
              '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        technologies.values(),
        labels=technologies.keys(),
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05),
        shadow=True
    )
    
    # Customize text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    for text in texts:
        text.set_fontsize(10)
        text.set_fontweight('bold')
    
    # Add title
    ax.set_title('Object Detection System - Technology Stack Distribution', 
                 fontsize=16, fontweight='bold', color='#2C3E50', pad=20)
    
    # Add subtitle
    ax.text(0, -1.3, 'Technology components and their contribution to the system', 
            ha="center", va="center", fontsize=12, style='italic', color="#7F8C8D")
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('technology_piechart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print("✅ Technology pie chart generated successfully!")
    print("📁 Saved as: technology_piechart.png")

def create_performance_chart():
    """Create a performance metrics bar chart"""
    
    # Performance metrics
    metrics = ['Inference Speed', 'Accuracy', 'Memory Efficiency', 'CPU Usage', 'FPS Performance']
    values = [95, 95, 85, 70, 90]  # Percentage values
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar chart
    bars = ax.barh(metrics, values, color=colors, edgecolor='#2C3E50', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(value + 1, bar.get_y() + bar.get_height()/2, f'{value}%',
                ha='left', va='center', fontweight='bold', fontsize=11)
    
    # Customize chart
    ax.set_xlim(0, 100)
    ax.set_xlabel('Performance Score (%)', fontsize=12, fontweight='bold', color='#2C3E50')
    ax.set_title('Object Detection System - Performance Metrics', 
                 fontsize=16, fontweight='bold', color='#2C3E50', pad=20)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Customize ticks
    ax.set_xticks(range(0, 101, 20))
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('performance_chart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print("✅ Performance chart generated successfully!")
    print("📁 Saved as: performance_chart.png")

def main():
    """Generate all visualizations"""
    print("🎨 Creating professional visualizations for Object Detection Project...")
    print("=" * 60)
    
    # Create all charts
    create_clean_flowchart()
    create_technology_piechart()
    create_performance_chart()
    
    print("=" * 60)
    print("🎉 All visualizations created successfully!")
    print("📊 Generated files:")
    print("   • clean_flowchart.png - Development pipeline")
    print("   • technology_piechart.png - Technology distribution")
    print("   • performance_chart.png - Performance metrics")

if __name__ == "__main__":
    main()
