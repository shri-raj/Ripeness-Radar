import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from keras.models import load_model

model = YOLO('models/yolov8m-seg.pt')

classifier = load_model('models/apple_classifier.h5')

def preprocess_for_classification(img):
    img = cv2.resize(img, (150, 150))
    img = img.astype(np.float32) / 255.0 
    img = np.expand_dims(img, axis=0)
    return img

def classify_apple(apple_img):
    processed_img = preprocess_for_classification(apple_img)
    preds = classifier.predict(processed_img)
    class_idx = np.argmax(preds)  
    return class_idx  

def detect_and_classify_apples(image):
    results = model(image)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            if class_id == 47:
                x1, y1, x2, y2 = box.xyxy[0].int()
                apple_img = image[int(y1):int(y2), int(x1):int(x2)]

                if apple_img.size == 0:
                    continue
                
                apple_class = classify_apple(apple_img)

                class_labels = {0: "Ripe", 1: "Rotten", 2: "Unripe"}
                label = class_labels.get(apple_class, "Unknown")
                
                red = green = 256
                if label == "Ripe":
                    red = 0
                elif label == "Rotten":
                    green = 0
                
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (red, green, 0), 2)
                cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (red, green, 0), 2)

    return image

iface = gr.Interface(
    fn=detect_and_classify_apples,
    inputs=gr.inputs.Image(type="numpy", label="Upload Image"),
    outputs=gr.outputs.Image(type="numpy", label="Detected Apples"),
    title="Apple Detection and Classification",
    description="Upload an image to detect and classify apples into Ripe, Unripe, or Rotten."
)

iface.launch()