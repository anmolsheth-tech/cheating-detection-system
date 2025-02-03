import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Try "yolov8s.pt" for better accuracy

# Allowed classes (only detecting mobile phones, books as chits, laptops)
allowed_classes = ["cell phone", "book", "laptop"]  # 'book' acts as a chit proxy

# Load the video file
video_path = "test-video.mp4"  # Change this to 0 for webcam
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object
output_path = "filtered_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ FIXED: Pass the frame instead of the video path
    results = model.predict(frame, conf=0.25)  

    # Process the detections
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())  # ✅ FIXED: Correctly extract class index
            class_name = model.names[class_id]  # ✅ FIXED: Properly retrieve class name

            if class_name in allowed_classes:  # Only keep cheating-related objects
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                confidence = float(box.conf[0])  # Confidence score

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the filtered detections
    cv2.imshow("Filtered Cheating Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
