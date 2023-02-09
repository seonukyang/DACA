import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(X_train, Y_train), (X_test, Y_test) = datasets.cifar10.load_data()

#독립변수 정규화
X_train = X_train/255
X_test = X_test/255



class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[Y_train[i][0]])
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

sol_history = model.fit(X_train, Y_train, epochs=20,
                    validation_data=(X_test, Y_test))
Y_pred = np.round(model.predict(X_test, verbose=0),3)
Y_classes = model.predict_classes(X_test, verbose=0)

plt.plot(sol_history.history['accuracy'], label='accuracy')
plt.plot(sol_history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.ylim([0.5, 1])
plt.legend(loc='lower right')



# --- predict all test image and find wrong predictions ---

# n_wrong = []
# n_right = []
# for i in range(10000):
#     if test_labels[i] != np.argmax(Y_pred[i]):
#         n_wrong.append(i)
#     else:
#         n_right.append(i)
# print(len(n_wrong))

# plt.figure(figsize=(5,5))
# for i in range(25):
#     n_img = n_wrong[i]
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     predict_image = class_names[np.argmax(Y_pred[n_img])]
#     plt.imshow(test_images[n_img], cmap=plt.cm.binary)
#     plt.title('(T):'+class_names[test_labels[n_img][0]]+'/(P):'+predict_image)
# plt.show()

# plt.figure(figsize=(5,5))
# for i in range(25):
#     n_img = n_right[i]
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     predict_image = class_names[np.argmax(Y_pred[n_img])]
#     plt.imshow(test_images[n_img], cmap=plt.cm.binary)
#     plt.title('(T):'+class_names[test_labels[n_img][0]]+'/(P):'+predict_image)
# plt.show()

train_score_cnn = model.evaluate(X_train, Y_train, verbose=0)
test_score_cnn = model.evaluate(X_test, Y_test, verbose=0)
print('cnn평가용데이터에대한 예측 클래스',Y_classes)
print('cnn학습용 데이터 세트 오차와 정확도',train_score_cnn[0], train_score_cnn[1])
print('cnn평가용 데이터 세트 오차와 정확도',test_score_cnn[0], test_score_cnn[1])
