import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create figure
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Define boxes (text, x, y)
boxes = [
    ("Dataset Collection\n(Custom / COCO)", 5, 9),
    ("Data Preprocessing\n(Resize, Label, Normalize)", 5, 7.8),
    ("Model Training\n(TensorFlow / YOLO / OpenCV)", 5, 6.6),
    ("Model Testing & Validation\n(Accuracy Evaluation)", 5, 5.4),
    ("Model Conversion\n(.tflite / Lightweight Model)", 5, 4.2),
    ("Raspberry Pi Setup\n(Library Install, Camera Config)", 5, 3.0),
    ("Real-time Detection\n(Live Object Recognition)", 5, 1.8),
    ("Performance Evaluation\n(Latency, FPS, Accuracy)", 5, 0.6)
]

# Draw rectangles and text
for text, x, y in boxes:
    rect = mpatches.FancyBboxPatch(
        (x - 3.3, y - 0.4), 6.6, 0.8,
        boxstyle="round,pad=0.05",
        facecolor="#A7C7E7", edgecolor="black", linewidth=1.5
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=10, fontweight="bold")

# Draw arrows
for i in range(len(boxes) - 1):
    _, x, y = boxes[i]
    _, x2, y2 = boxes[i + 1]
    ax.arrow(x, y - 0.45, 0, (y2 - y) + 0.4, head_width=0.2, head_length=0.2, fc="black", ec="black")

# Title
ax.text(5, 9.7, "Object Detection on Raspberry Pi - Flowchart", ha="center", va="center", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig('object_detection_flowchart.png', dpi=300, bbox_inches='tight')
plt.show()
print("Flowchart saved as 'object_detection_flowchart.png'")
