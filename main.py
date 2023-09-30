
from tensorflow import keras
from keras import datasets, layers, models, Sequential
import cv2
import numpy as np

(img_train, label_train),(img_test,label_test) = datasets.cifar10.load_data()
img_train, img_test = img_train/150.0 , img_test/150.0

model = Sequential([
  layers.Conv2D(16,(3,3),activation='relu',input_shape = (32,32,3)),
  layers.MaxPool2D(2,2),
  layers.Conv2D(32,(3,3),activation='relu'),
  layers.MaxPool2D(2,2),
  layers.Conv2D(64,(3,3),activation='relu'),
  layers.MaxPool2D(2,2),
  layers.Flatten(),
  layers.Dense(64,activation = 'relu'),
  layers.Dense(10,activation='softmax')
])

model.compile( optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(img_train,label_train,epochs = 10, validation_data = (img_test,label_test))


loss,accuracy = model.evaluate(img_test,label_test)

print(f'loss is {loss}')
print(f'accuracy is {accuracy}')

class_names = ['Plane','Car','Bird','Cat','Deer','Dog','Frog',' Horse','Ship','Truck']

img = cv2.imread('cat.jpg')

prediction = model.predict(np.array(img) / 150)
index = np.argmax(prediction)
print(f'the prediction is {class_names[index]}')

