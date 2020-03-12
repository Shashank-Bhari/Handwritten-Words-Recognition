import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import skimage.io
from skimage.feature import hog
import os


path='words1/train/'
hog_list=[]
labels=[]
f=open('pre.txt','+w')
for i in range(1,211):
    path1=path+str(i)
    for j in os.listdir(path1):
        img=skimage.io.imread(fname=path1+'/'+j,as_gray=True)
        fd=hog(img,orientations=9,pixels_per_cell=(14,14),cells_per_block=(3,3),transform_sqrt=True,block_norm="L1")
        print(path1+'/'+j)
        hog_list.append(fd)
        labels.append(i)


x_train,x_test,y_train,y_test=train_test_split(hog_list,labels,test_size=0.2,random_state=42)
for k in range(51,152,2):
	wkn=KNeighborsClassifier(n_neighbors=k,weights='distance')
	wkn.fit(x_train,y_train)
	
	pre=wkn.predict(x_test)
	f.write(str(k)+'-'+str(accuracy_score(y_test,pre))+"\n")
	print(str(k)+'-'+str(accuracy_score(y_test,pre))+"\n")        
f.close()

