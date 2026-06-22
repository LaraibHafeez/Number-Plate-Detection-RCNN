import cv2
import pandas as pd
import numpy as np
from sklearn.datasets import load_files
from LAB17_NMS_Iou import IoU

annotations = pd.read_csv(r"D:\PythonProject\numberplates\numberplates\annotations.csv")
allnames = annotations.iloc[:, [0]].values
box_list = annotations.iloc[:, [3,4,5,6]]
allnames = np.ndarray.flatten(allnames)
print(allnames)
print(box_list)

car_save_path="Used_Car/Plate/"
no_car_save_path="Used_Car/No_Plate/"
total_car=0
total_no_car=0

for i in range(len(allnames)):
    file=allnames[i]
    ss=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    img=cv2.imread(r"D:\PythonProject\numberplates\numberplates\\" + str(file))
    img_copy=img.copy()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    results=ss.process() #ROI(x1,y1,x2,y2)
    car_count=0
    no_car_count=0
    total_counted=0

    for box in results: #for every ROI
        found_box_use=[box[0],box[1],box[0]+box[2],box[1]+box[3]]
        image_roi=img[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
        iou=IoU(found_box_use,box_list.iloc[i].values)

        if iou>0.7:
            if car_count<16:
                image_roi_use=cv2.resize(image_roi,(128,128))
                image_roi_use=image_roi_use.reshape((128,128,3))
                cv2.imwrite(car_save_path+'Plate'+str(total_car)+'.jpg',image_roi_use)
                total_car+=1
                car_count+=1

        if iou<0.3:
            if no_car_count<16:
                image_roi_use=cv2.resize(image_roi,(128,128))
                image_roi_use=image_roi_use.reshape((128,128,3))
                cv2.imwrite(no_car_save_path+'no_Plate'+str(total_no_car)+'.jpg',image_roi_use)
                total_no_car+=1
                no_car_count+=1

        if total_counted>999:
            break

        total_counted+=1
        #32*50=1600
######################################################################

def load_data():
    data=load_files(r"D:\PythonProject\Used_Car")
    filenames=data['filenames']
    X=[]
    Y=[]
    not_count=0
    for name in filenames:
        if 'No_Plate' in name:
            not_count+=1
            if not_count<2300:
                img=cv2.imread(name)
                if img is not None:
                    X.append(img)
                    Y.append(0)
        else:
            img=cv2.imread(name)
            if img is not None:
                X.append(img)
                Y.append(1)
    X=np.asarray(X,dtype=object)
    Y=np.asarray(Y)
    return X,Y
