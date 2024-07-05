import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLOv10

# Load the YOLOv10 model
model = YOLOv10('best.pt')

st.title("Helmet Detection App")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def process_results(results):
    """
    Processes the detection results to extract bounding boxes, scores, class IDs, and class names.
    """
    # Ensure results is not a list
    if isinstance(results, list):
        results = results[0]

    boxes = results.boxes.xyxy.cpu().numpy()  # Bounding boxes
    scores = results.boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = results.boxes.cls.cpu().numpy().astype(int)  # Class IDs
    class_names = results.names  # Class names

    return boxes, scores, class_ids, class_names

def draw_boxes(image, boxes, scores, class_ids, class_names):
    """
    Draws bounding boxes and labels on the image.
    """
    # Define colors for classes
    colors = {'helmet': (0, 255, 0)}

    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        name = class_names[class_id]
        label = f"{name} {score:.2f}"
        color = colors.get(name, (255, 0, 0))  # Green for helmet, Red otherwise
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the uploaded image
    st.image(image_rgb, caption='Uploaded Image.', use_column_width=True)
    st.write("Detecting...")

    # Perform detection
    results = model(image_rgb)

    # Process results
    boxes, scores, class_ids, class_names = process_results(results)

    # Draw bounding boxes on the image
    image_rgb = draw_boxes(image_rgb, boxes, scores, class_ids, class_names)

    # Display the processed image
    st.image(image_rgb, caption='Processed Image', use_column_width=True)
