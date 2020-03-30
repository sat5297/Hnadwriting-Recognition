import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
             
model.fit(x_train,y_train,epochs=3)

val_loss,val_acc = model.evaluate(x_test,y_test)
print(val_loss,val_acc)

model.save('num_recogo.model')

new_model = tf.keras.models.load_model('num_recogo.model')

predictions = model.predict([x_test])

print(predictions)

import numpy as np
print(np.argmax(predictions[0]))

import matplotlib.pyplot as plt
plt.imshow(x_test[0],cmap = plt.cm.binary)
plt.show()


import os
predictions = model.predict([x_test])
p1 = [x_test[11],x_test[50],x_test[54],x_test[45],x_test[75],x_test[41],x_test[16]]
p1 = tf.keras.utils.normalize(p1,axis=1)
for ele in p1:
    plt.imshow(ele,cmap= plt.cm.binary)
    plt.show()


predict = model.predict([p1])


for ele1,ele2 in zip(predict,p1):
    print(np.argmax(ele1))
    plt.imshow(ele2,cmap=plt.cm.binary)
    plt.show()

