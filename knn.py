import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from skimage import feature
import imutils
import pickle



'''

def pre(img):
    im1=cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)




    mfl=cv2.medianBlur(im1,3)



    c=np.column_stack(np.where(mfl>0))
    angle=cv2.minAreaRect(c)[-1]
    if angle<=-45:
        angle=-(90+angle)
    else:
        angle=-angle

    (h,w)=mfl.shape[:2]
    center=(w//2,h//2)
    M=cv2.getRotationMatrix2D(center,angle,1.0)
    rotated=cv2.warpAffine(mfl,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

    img=rotated
    return img
'''
n=int(input())
if n==1:

    data=[]
    labels=[]
    path="/home/sooraj/withoutnoise/Words1/a"
    for i in range(15):
        for j in range(7):
            path1=path+str(i)+'_'+str(j)
            for k in os.listdir(path1):
                path2=path1+'/'+k
                print(path2)

                img=cv2.imread(path2)
                img=cv2.resize(img,(200,100))
                H=feature.hog(img,orientations=8,pixels_per_cell=(10,10),cells_per_block=(5,5),transform_sqrt=True,block_norm="L1")
                data.append(H)
                labels.append((i,j))

        model=KNeighborsClassifier(n_neighbors=43)
        model.fit(data,labels)
        pickle.dump(model,open("mod.sav","wb"))
else:
        model=pickle.load(open("mod.sav",'rb'))
        l=[]
        label=[]
        path="/home/sooraj/withoutnoise/tune/Words1/a"
        '''
        path1=path+'0_0'
        for k in os.listdir(path1):
            path2=path1+'/'+k
            print(path2)
            img=cv2.imread(path2)
            img=cv2.resize(img,(200,100))
            H=feature.hog(img,orientations=8,pixels_per_cell=(10,10),cells_per_block=(5,5),transform_sqrt=True,block_norm="L1")
            pred=model.predict(H.reshape(1,-1))[0]
            label.append(pred)
        count=0
        for i in label:
            if i[0]==0 and i[1]==0:
                count=count+1
        l.append(count/len(label))
        print(l)
        '''
        img=cv2.imread(path+"0_0/131w1.jpg")
        img=cv2.resize(img,(200,100))
        H=feature.hog(img,orientations=9,pixels_per_cell=(10,10),cells_per_block=(2,2),transform_sqrt=True,block_norm="L1")
        pred=model.predict(H.reshape(1,-1))[0]
        print(pred)
