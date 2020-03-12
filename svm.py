import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import cv2
from sklearn.metrics import classification_report
import numpy as np
import os
import random
from sklearn.metrics import classification_report


n=int(input())

if n==1:
    hog_list=[]
    labels=[]
    path='words1/train/'
    for i in range(1,211):
        l=os.listdir(path+str(i))
        if len(l)>160:
            l=random.sample(l,160)
        for j in os.listdir(path+str(i)):
            img=cv2.imread(path+str(i)+'/'+j,0)
            fd=hog(img,orientations=9,pixels_per_cell=(14,14),cells_per_block=(3,3),transform_sqrt=True,block_norm="L1")
            hog_list.append(fd)
            labels.append(i)
            print(i)

    hog_list=np.array(hog_list,'float64')
    labels=np.array(labels)
    svm=LinearSVC(C=10,dual=False)


    #clf = GridSearchCV(svm, {'C':[1, 10]})
    #clf.fit(hog_list,labels)
    svm.fit(hog_list,labels)
    joblib.dump(svm,'s.sav')

if n==2:
    hog_list=[]
    labels=[]
    path='words1/test/'
    for i in range(1,211):
        for j in os.listdir(path+str(i)):
                img=cv2.imread(path+str(i)+'/'+j,0)
                fd=hog(img,orientations=9,pixels_per_cell=(14,14),cells_per_block=(3,3),transform_sqrt=True,block_norm="L1")
                hog_list.append(fd)
                labels.append(i)
                print(i)
    svm=joblib.load('s.sav')
    c=0
    pre=svm.predict(hog_list)
    print(classification_report(y_true=labels,y_pred=pre))
