import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import data
fashion_training_df = pd.read_csv('input/fashion-mnist_train.csv')
fashion_test_df = pd.read_csv('input/fashion-mnist_test.csv')

# data visualization
training = np.array(fashion_training_df, dtype='float32')
testing = np.array(fashion_test_df, dtype='float32')

W_grid = 15
L_grid = 15
fig, axes = plt.subplots(W_grid, L_grid, figsize=(17, 17))
axes = axes.flatten()

# n_training = len(training)
# for i in range(0, W_grid * L_grid):
#     index = np.random.randint(0, n_training)
#     axes[i].imshow(training[index, 1:].reshape(28, 28), cmap='gray')
#     axes[i].axis('off')
# plt.tight_layout()
# plt.show()

# training the model
X_train = training[:, 1:] / 255
Y_train = training[:, 0]

X_test = testing[:, 1:] / 255
Y_test = testing[:, 0]

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=12345)

X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)
X_validate = X_val.reshape(X_val.shape[0], *(28, 28, 1))

X_test = X_test.reshape(-1, 28, 28, 1)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

cnn_model = Sequential()
cnn_model.add(Conv2D(32,3,3, input_shape=(28,28,1), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Flatten())

cnn_model.add(Dense(units=32, activation='relu'))
cnn_model.add(Dense(units=10, activation='sigmoid'))

cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
epochs = 50

cnn_model.fit(X_train, Y_train, epochs=epochs, batch_size=512, verbose=1, validation_data=(X_validate, Y_val))

# evaluate the model
evaluation = cnn_model.evaluate(X_test, Y_test)
print('Test Accuracy: {:.3f}'.format(evaluation[1]))

predictions = cnn_model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
L =5
W =5
fig, axes = plt.subplots(L, W, figsize=(12, 12))
axes = axes.ravel()
for i in range(L*W):
    axes[i].imshow(X_test[i].reshape(28, 28))
    axes[i].set_title(f'Prediction: {predicted_classes[i]} True: {Y_test[i]}')
    axes[i].axis('off')
# plt.subplots_adjust(wspace=0.5)
# plt.show()

from sklearn.metrics import confusion_matrix
cn = confusion_matrix(Y_test, predicted_classes)
plt.figure(figsize=(14, 10))
sns.heatmap(cn, annot=True)
plt.show()