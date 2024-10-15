

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

# face embeddings loaded
print("[INFO ] loads face embeddings!!!")
data=pickle.loads(open("output/embeddings.pickle","rb").read())


# labels are encoded

print("[INFO ] encoding labels!!!")
le=LabelEncoder()
labels=le.fit_transform(data["names"])


# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition

print("[INFO ] models are trained!!!")
recognizer=SVC(C=1.0,kernel="linear",probability=True)
recognizer.fit(data["embeddings"],labels)


# actual face recognition model is written to disk

f=open("output/recognizer","wb")
f.write(pickle.dumps(recognizer))
f.close()

# label encoder is written to disk
f=open("output/le.pickle","wb")
f.write(pickle.dumps(le))
f.close()