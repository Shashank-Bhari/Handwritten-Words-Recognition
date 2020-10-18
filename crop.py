import cv2
import numpy as np
import os
import math
import imutils


def deskew(img):

    co=np.column_stack(np.where(img>0))
    x,y,w,h=cv2.boundingRect(co)
    img=img[x:x+w,y:y+h]
    img=crop(img)

    co=np.column_stack(np.where(img>0))
    rect=cv2.minAreaRect(co)
    angle=rect[-1]
    #box=cv2.boxPoints(rect)
    #box=np.int0(box)
    #cv2.drawContours(img,[box],0,255,2)
    #cv2.imshow('a',img)
    #cv2.waitKey(0)

    if angle<-45:
        angle=-(90+angle)
    else:
        angle=-angle
    print(angle)
    '''
    (h,w)=img.shape[:2]
    center=(w//2,h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    '''
    rotated=imutils.rotate(img,angle)
    #co=np.column_stack(np.where(rotated>0))

    return rotated





def crop(img):
    '''
    co=np.column_stack(np.where(img>0))
    x,y,w,h=cv2.boundingRect(co)
    img=img[x:x+w,y:y+h]
    '''
    y_sum=cv2.reduce(img,0,cv2.REDUCE_SUM,dtype=cv2.CV_32S)[0]
    i=0
    j=img.shape[1]-1
    y_norm=[(x)/max(y_sum) for x in y_sum]
    while(y_norm[i]<=0.1 and i<j):
        i=i+1
    while(y_norm[j]<=0.1 and j>0):
        j=j-1
    x_sum=cv2.reduce(img,1,cv2.REDUCE_SUM,dtype=cv2.CV_32S)
    x_norm=[(x)/max(x_sum) for x in x_sum]
    k=0
    l=img.shape[0]-1
    while(x_norm[k][0]<=0.1 and k<l):
        k=k+1
    while(x_norm[l][0]<=0.2 and l>0):
        l=l-1
    img1=img[k:l,i:j]
    #cv2.imshow("a",resize(img1,500,500))
    #cv2.waitKey(0)
    return img1

'''
def main():

    path='words/'
    dst='crop/'
    for i in range(1,211):
        try:
            os.mkdir(dst+str(i))
        except FileExistsError:
            pass
        for j in os.listdir(path+str(i)):
            img=cv2.imread(path+str(i)+'/'+j,0)
            im1=crop(img)
            im1=resize(im1)
            im1=open_close(im1)
            print(dst+str(i)+'/'+j)
            cv2.imwrite(dst+str(i)+'/'+j+'.jpg',im1)
'''
def one(img):
    img=trim_90(img)
    '''
    if(img.shape[0]<1000 and img.shape[1]<1000):
        img=cv2.resize(img,(1000,2000))
    if(img.shape[0]<1500 and img.shape[1]<1500):
        #img=cv2.resize(img,(1000,2000))
        img=resize(img,3000,4000)
    '''
    #img=cv2.bilateralFilter(img,9,75,75)

    #img=cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,71,20)
    #cv2.imshow('a',resize(img,500,500))
    #cv2.waitKey(0)
    #img=binary(img)
    #img=cv2.medianBlur(img,3)
    #img=open_close(img)
    #im1=crop(im1)
    im1=deskew(img)
    #im1=auto_crop(im1)
    #im1=crop(im1)
    im1=resize(im1,200,100)
    return im1


def resize(img,w,h):
    if(img.shape[0]>h):
        img1=cv2.resize(img,(img.shape[1],h),interpolation=cv2.INTER_AREA)
    else:
        img1=cv2.resize(img,(img.shape[1],h),interpolation=cv2.INTER_CUBIC)
    if(img.shape[1]>w):
        img1=cv2.resize(img1,(w,h),interpolation=cv2.INTER_AREA)
    else:
        img1=cv2.resize(img1,(w,h),interpolation=cv2.INTER_CUBIC)
    return img1



def open_close(img):
    #if(img.shape[0]<1000 and img.shape[1]<2000):
        #img=cv2.resize(img,(1000,2000))

    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    open=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    close=cv2.morphologyEx(open,cv2.MORPH_CLOSE,kernel)
    #cv2.imshow("a",resize(close,500,500))
    #cv2.waitKey(0)
    return close
'''
def remove(img):
    laplacian = cv2.Laplacian(img,cv2.CV_8UC1) # Laplacian Edge Detection
    minLineLength = 10
    maxLineGap = 10
    lines = cv2.HoughLinesP(laplacian,1,np.pi/180,100,minLineLength,maxLineGap)
    print(len(lines))
    if len(lines)<=0:
        return img

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),(0,0,0),1)

    return img
'''



def binary(img):
    edge=cv2.Canny(img.copy(),20,50)
    cont,hier=cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    '''
    #print(cont)
    for c in cont:

        x,y,w,h=cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
    cv2.imshow('abc',resize(img,700,700))
    cv2.waitKey(0)
    '''


    print(len(cont))
    can=np.zeros_like(img)
    #cv2.drawContours(can,cont,-1,255,20)
    img1=img
    for i in cont:
        x,y,w,h=cv2.boundingRect(i)
        can[y:y+h,x:x+w]=img[y:y+h,x:x+w]
        #cv2.imshow("a",resize(img[x:y,x+w:y+h],500,500))
        #cv2.waitKey(0)
    return can

def trim_90(img):
    a,b,c,d=0,img.shape[0]-1,0,img.shape[1]-1
    while(img[a].sum()==0 and a<img.shape[1]):
        a=a+1
    while(img[a].sum()==0 and b>0):
        b=b-1
    x_sum=cv2.reduce(img,0,cv2.REDUCE_SUM,dtype=cv2.CV_32S)[0]
    while(x_sum[c]==0 and a<img.shape[0]):
        c=c+1
    while(x_sum[d]==0 and d>0):
        d=d-1

    return img[a:b,c:d]

if __name__=='__main__':
    img=cv2.imread("abc.jpg",0)
    #if(img.shape[0]<1000 and img.shape[1]<2000):
        #img=resize(img,1000,2000)
    #img=cv2.bilateralFilter(img,9,75,75)
    #img=imutils.rotate_bound(img,90)
    #img=cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,71,20)
    #img=binary(img)

    cv2.imshow('a',resize(img,700,700))
    cv2.waitKey(0)

    img=one(img)

    cv2.imshow('a',img)
    cv2.waitKey(0)
