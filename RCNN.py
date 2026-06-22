import cv2
import matplotlib.pyplot as plt
import numpy as np
from LAB17_NMS_Iou import NMS, IoU
from LAB_18_READDATA import load_data
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from keras.optimizers import Adam
from keras.models import load_model, Sequential
input_shape = (128,128,3)
model=Sequential()
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
X,Y=load_data()
X=np.asarray(X,dtype=np.float32)
Y=np.asarray(Y,dtype=np.float32)
X=X/255.0
print(X.shape)
print(Y.shape)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
op=Adam(learning_rate=0.001)
model.compile(optimizer=op,loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(X_train,Y_train,epochs=20,validation_data=(X_test,Y_test))
print(model.evaluate(X_test,Y_test))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train Loss','Validation Loss'])
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train Accuracy','Validation Accuracy'])
plt.show()
model.save('Base_model.keras')
######################################################################
def rcnn(image,model):
    ss=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    results=ss.process()
    print("No. of RoI in test image=",len(results))
    copy1=image.copy()
    copy2=image.copy()
    positive_boxes=[]
    probs=[]
    for box in results:
        x1=box[0]
        y1=box[1]
        x2=box[0]+box[2]
        y2=box[1]+box[3]
        w=x2-x1
        h=y2-y1
        if w<60 or h<20:
            continue
        if w>320 or h>110:
            continue
        if w/h<2.5 or w/h>7.5:
            continue
        if y1<100:
            continue
        roi=image.copy()[y1:y2,x1:x2]
        if roi.size==0:
            continue
        roi=cv2.resize(roi,(128,128))
        roi=roi.astype(np.float32)/255.0
        roi_use=roi.reshape((1,128,128,3))
        class_pred=model.predict(roi_use,verbose=0)[0][0]
        if class_pred > 0.85:
            positive_boxes.append([x1,y1,x2,y2])
            probs.append(class_pred)
            cv2.rectangle(copy2,(x1,y1),(x2,y2),(0,255,0),2)
    if len(positive_boxes) > 0:
        cleaned_boxes = NMS(np.array(positive_boxes), 0.2)
        cleaned_boxes = np.array(cleaned_boxes, dtype='int')
        best_box = None
        best_area = 999999
        for cleaned_box in cleaned_boxes:
            x1, y1, x2, y2 = cleaned_box
            w = x2 - x1
            h = y2 - y1
            area = w * h
            if area < best_area:
                best_area = area
                best_box = cleaned_box
        if best_box is not None:
            cv2.rectangle(copy1, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (255, 0, 0), 4)
    plt.imshow(cv2.cvtColor(copy1,cv2.COLOR_BGR2RGB))
    plt.title("After NMS - copy1")
    plt.show()
    plt.imshow(cv2.cvtColor(copy2,cv2.COLOR_BGR2RGB))
    plt.title("Before NMS - copy2")
    plt.show()
######################################################################
rcnn_model=tf.keras.models.load_model('Base_model.keras')
for img_name in ['test_img1.jpg','test_img2.jpg']:
    test_img=cv2.imread(img_name)
    test_img=cv2.resize(test_img,(400,300))
    print("Testing image:",img_name)

    rcnn(test_img,rcnn_model)

######################################################################00
