from imutils import paths
import numpy as np
import imutils
import argparse
import cv2
import os
import pickle

print("[INFO] loading face detector...")
protoPath="face_detection_model/deploy.prototxt.txt"
modelPath="face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNet(protoPath, modelPath)

print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNet(protoPath, modelPath)

# paths to the input images in our dataset are grabbed
print("[INFO] quantifying faces...")
imagePaths=list(paths.list_images('dataset'))

# the lists of extracted facial embeddings and corresponding people names are initialized
knownEmbeddings=[]
knownNames=[]

#  total number of faces processed are initialized
total=0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    if (i%50==0):
        print("Processing image {}/{}".format(i, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                          (300, 300), (104.0, 177.0, 123.0),
                                          swapRB=False, crop=False)


        detector.setInput(imageBlob)
        detections = detector.forward()

        if len(detections) > 0:
            i=np.argmax(detections[0,0,:,2])
            confidence=detections[0,0,i,2]


            if confidence > 0.5:


                box = detections[0,0,i,3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
# face ROI is extracted and ROI dimensions are grabbed
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

# ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
                faceBlob=cv2.dnn.blobFromImage(face,1.0/255,
                     (96,96),(0,0,0),swapRB=True,crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

            #  name of the person + corresponding face embedding are added to their respective lists
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total+=1

            # facial embeddings + names are dumped to disk
print("[INFO] seralizing {} encodings...".format(total))
data={"embeddings":knownEmbeddings,"names":knownNames}
f=open("output/embeddings.pickle","wb")
f.write(pickle.dumps(data))
f.close()