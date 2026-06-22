import numpy as np
import cv2
from sklearn.datasets import load_files

def NMS(boxes, overlapThreshold):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == 'i':
        boxes = boxes.astype('float')
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idx = np.argsort(y2)
    while len(idx) > 0:
        last = len(idx) - 1
        i = idx[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idx[:last]])
        yy1 = np.maximum(y1[i], y1[idx[:last]])
        xx2 = np.minimum(x2[i], x2[idx[:last]])
        yy2 = np.minimum(y2[i], y2[idx[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idx[:last]]
        idx = np.delete(idx, np.concatenate(([last], np.where(overlap > overlapThreshold)[0])))
    return boxes[pick].astype(int)

#######################################################################
def IoU(boxA,boxB):
        xA=max(boxA[0],boxB[0])  #left
        yA=max(boxA[1],boxB[1]) #top
        xB=min(boxA[2],boxB[2]) #ri8
        yB=min(boxA[3],boxB[3])  #bottom
        interW=max(0,xB-xA)
        interH=max(0,yB-yA)
        interArea=interW*interH
        boxAArea=max(0,(boxA[2]-boxA[0]))*max(0,(boxA[3]-boxA[1]))
        boxBArea=max(0,(boxB[2]-boxB[0]))*max(0,(boxB[3]-boxB[1]))
        denominator=float(boxAArea + boxBArea - interArea)
        if denominator==0:
            return 0
        iou = interArea / denominator
        return iou

##################################
def load_data():
    data=load_files(r'D:\PythonProject\Used_Car')
    filenames=data['filenames']
    X=[]
    Y=[]
    not_count=0
    for name in filenames:
        if 'No_Plate' in name or 'No_Plate'.lower() in name.lower():
            not_count+=1
            if not_count<2300:
                img=cv2.imread(name)
                if img is not None:
                    X.append(img)
                    Y.append(0)
            else:
                pass
        else:
            img=cv2.imread(name)
            if img is not None:
                X.append(img)
                Y.append(1)
    X=np.asarray(X,dtype=object) #diff size
    Y=np.asarray(Y)
    return X,Y
