import joblib
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import cv2
import numpy as np
import os
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

hog_list=[]
labels=[]
path='words1/train/'
for i in range(1,211):
    dir=os.listdir(path+str(i))
    if len(dir)>160:
        dir=random.sample(dir,160)
    for j in dir:
        img=cv2.imread(path+str(i)+'/'+j,0)
        fd=hog(img,orientations=8,pixels_per_cell=(10,10),cells_per_block=(5,5),transform_sqrt=True,block_norm="L1")
        hog_list.append(fd)
        labels.append(i)
        print(i)

hog_list=np.array(hog_list,'float64')
labels=np.array(labels)


rf=RandomForestClassifier()


clf=GridSearchCV(rf,param_grid={'n_estimators':[100,200],'min_samples_leaf':[2,3]})
clf.fit(hog_list,labels)

hog_list=[]
labels=[]
path='words1/test/'
for i in range(1,211):
    dir=os.listdir(path+str(i))
    if len(dir)>160:
        dir=random.sample(dir,160)
    for j in dir:
        img=cv2.imread(path+str(i)+'/'+j,0)
        fd=hog(img,orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),transform_sqrt=True,block_norm="L1")
        hog_list.append(fd)
        labels.append(i)
        print(i)

hog_list=np.array(hog_list,'float64')
labels=np.array(labels)


pred=clf.predict(hog_list)

print(classification_report(y_true=labels,y_pred=pred))
