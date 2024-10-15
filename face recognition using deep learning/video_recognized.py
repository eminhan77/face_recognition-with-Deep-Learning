import numpy as np
import cv2
import imutils
import os
import time
import pickle
from imutils.video import VideoStream
from imutils.video import FPS

print("Face Detecor loaded$$$")
protoPath="face_detection_model/deploy.prototxt.txt"
modelPath="face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNet(protoPath, modelPath)

print("Face Recognizer loaded$$$")
embedder=cv2.dnn.readNetFromTorch(protoPath, modelPath)

recognizer=pickle.load(open("recognizer.pkl", "rb"))
le=pickle.loads(open("output/le.pkl", "rb"))

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

fps = FPS().start()

while True:
    frame = vs.read()

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(
             cv2.resize(frame,(300,300)), 1.0, (300, 300),
         (104.0,177.0,123.0), swapRB=False,crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

                faceBlob=cv2.dnn.blobFromImage(face, 1.0/255,
                    (96,96), (0,0,0), swapRB=True,crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes[j]

                text="{}:{:.2f}%".format(name,proba*100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, test, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                fps.update()
                cv2.imshow("Frame",frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                    # timer stopped and fps information displayed
                    fps.stop()
                    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
                    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

                    cv2.destroyAllWindows()
                    vs.stop()

