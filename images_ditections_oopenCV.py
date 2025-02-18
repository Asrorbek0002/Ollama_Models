# import cv2
# import cvlib as cv
# from cvlib.object_detection import draw_bbox
# from gtts import gTTS
# from playsound import playsound
#
#
# video=cv2.VideoCapture(0)
#
# while True:
#     ret, frame =video.read()
#     bbox, label, conf = cv.detect_common_objects(frame)
#     output_image = draw_bbox(frame, bbox, label, conf)
#
#     cv2.imshow("Object Detection", output_image)
#
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break


import cv2
import ollama
import time
import numpy as np
from pprint import pprint
import cvlib as cv
from cvlib.object_detection import draw_bbox
start_time = time.time()


yolo_weights = "C:/Users/Vivtus/Yolo_Files/yolov3.weights"
yolo_cfg = "C:/Users/Vivtus/Yolo_Files/yolov3.cfg"
yolo_labels = "C:/Users/Vivtus/Yolo_Files/coco.names"


# Initialize webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    bbox, label, conf = cv.detect_common_objects(frame, confidence=0.5, model='yolov3', enable_gpu=False)
    output_image = draw_bbox(frame, bbox, label, conf)

    if not ret:
        print("Failed to grab frame")
        break

    # Display the webcam feed
    cv2.imshow("Webcam Feed", output_image)

    # Allow the user to press 'q' to exit the loop
    key = cv2.waitKey(1) & 0xFF
    key=input("if you want to start 'y' aks holda 'n'>>> ")
    if key=='n':
    #if key == ord('q'):
        break

    # Start time for LLaVA model request


    # Convert the frame to PNG format in memory (not saving to disk)
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()

    # Perform LLaVA model analysis using the frame
    res = ollama.chat(
        model='llava',
        messages=[
            {'role': 'user',
             'content': 'Analyze this picture: size and calories, and provide exact numbers.',
             'images': [img_bytes]
             }
        ]
    )

    # End time for LLaVA model request
    end_time = time.time()

    # Print the analysis result
    pprint(res['message']['content'])

    # Print time taken for LLaVA analysis
    print("Exact time:", end_time - start_time, "seconds")

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()







