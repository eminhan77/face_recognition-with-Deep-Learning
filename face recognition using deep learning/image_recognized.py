
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# argument parser constructed and arguments are parsed
ap=argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())


# face detector is loaded from the disk
print("[INFO] loading face detector...")
protoPath=os.path.sep.join(['face_detection_models', 'deploy.prototxt'])
modelPath=os.pat.sep.join(['face_detection_models', 'res10_300x300_ssd_iter_140000.caffemodel'])
detector = cv2.dnn.readNet(protoPath, modelPath)


print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNet(protoPath, modelPath)

# actual face recognition model along with the label encoder loaded
recognizer=pickle.loads(open('output/recognizer.pickle',"rb").read())
le=pickle.loads(open('output/le.pickle',"rb").read())

    # image loaded
image=cv2.imread(args["image"])
image=imutils.resize(image,width=600)
(h, w) = image.shape[:2]

# blob from the image is constructed
imageBlob=cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)), 1.0, (300, 300),
    (104.0, 177.0, 123.0), swapRB=False, crop=False
)

detector.setInput(imageBlob)
detections=detector.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    #  confidence (i.e., probability) associated with the prediction is extracted
    confidence=detections[0, 0, i, 2]

    # filter out weak detections
    if confidence > 0.5:
        # (x, y)-coordinates of the bounding box for the face is computed
        box=detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

# face ROI is extracted
        face=image[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]

        # ensure the face width and height are sufficiently large
        if fW<20 or fH<20:
            continue

# construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
            faceBlob=cv2.dnn.blobFromImage(face,1.0/255,(96,96),(0,0,0),swapRB=True,crop=False)
            embedder.setInput(faceBlob)
            vec=embedder.forward()

# classification is performed to recognize the face
            preds=recognizer.predict(vec)[0]
            j=np.argmax(preds)
            proba=preds[j]
            name=le.classes[j]

# bounding box of the face along with the associated probability is drawn
            text="{}: {:.2f}%".format(name,proba*100)
            y=startY-10 if startY-10>10 else startY+10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


            # Output Image is showed
            cv2.imshow("Image", image)
            cv2.waitKey(0)