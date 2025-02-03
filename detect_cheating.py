import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Using YOLOv8n for faster detection

# Allowed classes (we only want to detect these)
allowed_classes = ["cell phone", "book", "laptop"]  # 'book' can be a proxy for chits

# Load the video file or use webcam
video_path = "test-video.mp4"  # Change this to 0 for a webcam
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

    # Run YOLO on the current frame
    results = model(frame)

    # Get detections and filter them
    filtered_boxes = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]  # Get object name
            
            if class_name in allowed_classes:  # Only keep mobile phones & chits
                filtered_boxes.append(box)

    # Draw filtered detections on frame
    for box in filtered_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
        confidence = float(box.conf[0])  # Confidence score
        class_id = int(box.cls[0])
        class_name = result.names[class_id]  # Get class name
        
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
