#%%
import cv2
import numpy
from numpy import expand_dims
from scipy.io import loadmat
import keras
from keras import layers
from keras.utils import np_utils
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from matplotlib import pyplot
from matplotlib.pyplot import figure
from skimage.transform import pyramid_gaussian
#%%
Train = loadmat('train_32x32.mat')
Test = loadmat('test_32x32.mat')
X_train = Train['X']
train_labels = Train['y']
X_test = Test['X']
test_labels = Test['y']
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
X_train = X_train[numpy.newaxis, ...]
X_train = numpy.swapaxes(X_train, 0, 4).squeeze()
X_test = X_test[numpy.newaxis, ...]
X_test = numpy.swapaxes(X_test, 0, 4).squeeze()
train_labels[train_labels == 10] = 0
test_labels[test_labels == 10] = 0
Y_train = np_utils.to_categorical(train_labels)
Y_test = np_utils.to_categorical(test_labels)
#%%
input = layers.Input(shape=(32, 32, 3))
conv1 = layers.Conv2D(32, 5, activation='relu', padding='same')(input)
batch1 = layers.BatchNormalization()(conv1)
pool1 = layers.MaxPool2D(pool_size=2)(batch1)
conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
batch2 = layers.BatchNormalization()(conv2)
pool2 = layers.MaxPool2D(pool_size=2)(batch2)
flatten = layers.Flatten()(conv2)
batchf = layers.BatchNormalization()(flatten)
fc1 = layers.Dense(100, activation='relu')(batchf)
batchfc1 = layers.BatchNormalization()(fc1)
output = layers.Dense(10, activation='softmax')(batchfc1)
model1 = Model(input, output)
model1.summary()
model1.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
#%%
train_output = model1.fit(X_train, Y_train, batch_size=128, epochs=20, validation_split=0.2)
model1.save('model1')
#%%
input = layers.Input(shape=(32, 32, 3))
conv1 = layers.Conv2D(32, 5, activation='relu', padding='same')(input)
batch1 = layers.BatchNormalization()(conv1)
pool1 = layers.MaxPool2D(pool_size=2)(batch1)
conv2 = layers.Conv2D(64, 5, activation='relu', padding='same')(pool1)
batch2 = layers.BatchNormalization()(conv2)
pool2 = layers.MaxPool2D(pool_size=2)(batch2)
conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
batch3 = layers.BatchNormalization()(conv3)
pool3 = layers.MaxPool2D(pool_size=2)(batch3)
flatten = layers.Flatten()(pool3)
batchf = layers.BatchNormalization()(flatten)
fc1 = layers.Dense(100, activation='relu')(batchf)
batchfc1 = layers.BatchNormalization()(fc1)
output = layers.Dense(10, activation='softmax')(batchfc1)
model2 = Model(input, output)
model2.summary()
model2.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
#%%
train_output = model2.fit(X_train, Y_train, batch_size=128, epochs=20, validation_split=0.2)
model2.save('model2')
#%%
input = layers.Input(shape=(32, 32, 3))
conv1 = layers.Conv2D(32, 3, activation='relu')(input)
batch1 = layers.BatchNormalization()(conv1)
conv2 = layers.Conv2D(32, 3, activation='relu', padding='same')(batch1)
batch2 = layers.BatchNormalization()(conv2)
pool2 = layers.MaxPool2D(pool_size=2)(batch2)
conv3 = layers.Conv2D(64, 3, activation='relu')(pool2)
batch3 = layers.BatchNormalization()(conv3)
conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(batch3)
batch4 = layers.BatchNormalization()(conv4)
pool4 = layers.MaxPool2D(pool_size=2)(batch4)
flatten = layers.Flatten()(pool4)
batchf = layers.BatchNormalization()(flatten)
fc1 = layers.Dense(512, activation='relu')(batchf)
batchfc1 = layers.BatchNormalization()(fc1)
output = layers.Dense(10, activation='softmax')(batchfc1)
model3 = Model(input, output)
model3.summary()
model3.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
#%%
train_output = model3.fit(X_train, Y_train, batch_size=128, epochs=20, validation_split=0.2)
model3.save('model3')
#%%
input = layers.Input(shape=(32, 32, 3))
conv1 = layers.Conv2D(32, 3, activation='relu')(input)
batch1 = layers.BatchNormalization()(conv1)
conv2 = layers.Conv2D(32, 3, activation='relu', padding='same')(batch1)
batch2 = layers.BatchNormalization()(conv2)
pool2 = layers.MaxPool2D(pool_size=2)(batch2)
conv3 = layers.Conv2D(64, 3, activation='relu')(pool2)
batch3 = layers.BatchNormalization()(conv3)
conv4 = layers.Conv2D(64, 3, activation='relu', padding='same')(batch3)
batch4 = layers.BatchNormalization()(conv4)
pool4 = layers.MaxPool2D(pool_size=2)(batch4)
conv5 = layers.Conv2D(128, 3, activation='relu')(pool4)
batch5 = layers.BatchNormalization()(conv5)
pool5 = layers.MaxPool2D(pool_size=2)(batch5)
flatten = layers.Flatten()(pool5)
batchf = layers.BatchNormalization()(flatten)
fc1 = layers.Dense(512, activation='relu')(batchf)
batchfc1 = layers.BatchNormalization()(fc1)
output = layers.Dense(10, activation='softmax')(batchfc1)
model4 = Model(input, output)
model4.summary()
model4.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
#%%
train_output = model4.fit(X_train, Y_train, batch_size=128, epochs=20, validation_split=0.2)
model4.save('model4')
#%%
input = layers.Input(shape=(32, 32, 3))
conv1 = layers.Conv2D(32, 3, activation='relu')(input)
batch1 = layers.BatchNormalization()(conv1)
conv2 = layers.Conv2D(32, 3, activation='relu', padding='same')(batch1)
batch2 = layers.BatchNormalization()(conv2)
pool2 = layers.MaxPool2D(pool_size=2)(batch2)
conv3 = layers.Conv2D(64, 3, activation='relu')(pool2)
batch3 = layers.BatchNormalization()(conv3)
conv4 = layers.Conv2D(64, 3, activation='relu', padding='same')(batch3)
batch4 = layers.BatchNormalization()(conv4)
pool4 = layers.MaxPool2D(pool_size=2)(batch4)
conv5 = layers.Conv2D(128, 3, activation='relu')(pool4)
batch5 = layers.BatchNormalization()(conv5)
conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(batch5)
batch6 = layers.BatchNormalization()(conv6)
pool6 = layers.MaxPool2D(pool_size=2)(batch6)
flatten = layers.Flatten()(pool6)
batch7 = layers.BatchNormalization()(flatten)
flat2 = layers.Dense(512, activation='relu')(batch7)
batch8 = layers.BatchNormalization()(flat2)
output = layers.Dense(10, activation='softmax')(batch8)
model5 = Model(input, output)
model5.summary()
model5.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
#%%
train_output = model5.fit(X_train, Y_train, batch_size=128, epochs=20, validation_split=0.2)
model5.save('model5')
#%%
test_loss, test_acc = model1.evaluate(X_test, Y_test)
print('test loss = ', test_loss, 'test accuracy = ', test_acc)
test_loss, test_acc = model2.evaluate(X_test, Y_test)
print('test loss = ', test_loss, 'test accuracy = ', test_acc)
test_loss, test_acc = model3.evaluate(X_test, Y_test)
print('test loss = ', test_loss, 'test accuracy = ', test_acc)
test_loss, test_acc = model4.evaluate(X_test, Y_test)
print('test loss = ', test_loss, 'test accuracy = ', test_acc)
test_loss, test_acc = model5.evaluate(X_test, Y_test)
print('test loss = ', test_loss, 'test accuracy = ', test_acc)
#%%
input = layers.Input(shape=(32, 32, 3))
conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(input)
batch1 = layers.BatchNormalization()(conv1)
conv2 = layers.Conv2D(32, 3, activation='relu', padding='same')(batch1)
batch2 = layers.BatchNormalization()(conv2)
pool2 = layers.MaxPool2D(pool_size=2)(batch2)
drop2 =  layers.Dropout(0.25)(pool2)
conv3 = layers.Conv2D(64, 3, activation='relu')(drop2)
batch3 = layers.BatchNormalization()(conv3)
conv4 = layers.Conv2D(64, 3, activation='relu', padding='same')(batch3)
batch4 = layers.BatchNormalization()(conv4)
pool4 = layers.MaxPool2D(pool_size=2)(batch4)
drop4 =  layers.Dropout(0.25)(pool4)
conv5 = layers.Conv2D(128, 3, activation='relu')(drop4)
batch5 = layers.BatchNormalization()(conv5)
conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(batch5)
batch6 = layers.BatchNormalization()(conv6)
pool6 = layers.MaxPool2D(pool_size=2)(batch6)
drop6 = layers.Dropout(0.25)(pool6)
flatten = layers.Flatten()(drop6)
batchf = layers.BatchNormalization()(flatten)
fc1 = layers.Dense(512, activation='relu')(batchf)
batchfc1 = layers.BatchNormalization()(fc1)
dropfc1 =  layers.Dropout(0.3)(batchfc1)
output = layers.Dense(10, activation='softmax')(dropfc1)
best_model = Model(input, output)
best_model.summary()
best_model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
#%%
train_output = best_model.fit(X_train, Y_train, batch_size=128, epochs=20)
best_model.save('best_model')
#%%
test_loss, test_acc = best_model.evaluate(X_test, Y_test)
print('test loss = ', test_loss, 'test accuracy = ', test_acc)
#%%
x_train = X_train.reshape((len(X_train), numpy.prod(X_train.shape[1:])))
x_test = X_test.reshape((len(X_test), numpy.prod(X_test.shape[1:])))
#%%
modelf1 = Sequential()
modelf1.add(Dense(4000, activation='relu', input_shape=(numpy.prod(X_test.shape[1:]),)))
modelf1.add(Dropout(0.3))
modelf1.add(Dense(2000, activation='relu'))
modelf1.add(Dropout(0.3))
modelf1.add(Dense(10, activation='softmax'))
modelf1.summary()
modelf1.compile(optimizer=SGD(lr=0.001,decay=1e-7, momentum=0.9, nesterov=True), loss=categorical_crossentropy, metrics=['accuracy'])
#%%
modelf1.fit(x_train, Y_train, batch_size=128, epochs=20, validation_split=0.2)
#%%
modelf2 = Sequential()
modelf2.add(Dense(4000, activation='relu', input_shape=(numpy.prod(X_test.shape[1:]),)))
modelf2.add(Dropout(0.3))
modelf2.add(Dense(2000, activation='relu'))
modelf2.add(Dropout(0.3))
modelf2.add(Dense(1000, activation='relu'))
modelf2.add(Dropout(0.3))
modelf2.add(Dense(10, activation='softmax'))
modelf2.summary()
modelf2.compile(optimizer=SGD(lr=0.001,decay=1e-7, momentum=0.9, nesterov=True), loss=categorical_crossentropy, metrics=['accuracy'])
#%%
modelf2.fit(x_train, Y_train, batch_size=128, epochs=20, validation_split=0.2)
#%%
modelf3 = Sequential()
modelf3.add(Dense(4000, activation='relu', input_shape=(numpy.prod(X_test.shape[1:]),)))
modelf3.add(Dropout(0.3))
modelf3.add(Dense(2000, activation='relu'))
modelf3.add(Dropout(0.3))
modelf3.add(Dense(1000, activation='relu'))
modelf3.add(Dropout(0.3))
modelf3.add(Dense(500, activation='relu'))
modelf3.add(Dropout(0.3))
modelf3.add(Dense(10, activation='softmax'))
modelf3.summary()
modelf3.compile(optimizer=SGD(lr=0.001,decay=1e-7, momentum=0.9, nesterov=True), loss=categorical_crossentropy, metrics=['accuracy'])
#%%
modelf3.fit(x_train, Y_train, batch_size=128, epochs=20, validation_split=0.2)
#%%
modelf4 = Sequential()
modelf4.add(Dense(4000, activation='relu', input_shape=(numpy.prod(X_test.shape[1:]),)))
modelf4.add(Dropout(0.3))
modelf4.add(Dense(2000, activation='relu'))
modelf4.add(Dropout(0.3))
modelf4.add(Dense(1000, activation='relu'))
modelf4.add(Dropout(0.3))
modelf4.add(Dense(500, activation='relu'))
modelf4.add(Dropout(0.3))
modelf4.add(Dense(250, activation='relu'))
modelf4.add(Dropout(0.3))
modelf4.add(Dense(10, activation='softmax'))
modelf4.summary()
modelf4.compile(optimizer=SGD(lr=0.001,decay=1e-7, momentum=0.9, nesterov=True), loss=categorical_crossentropy, metrics=['accuracy'])
#%%
modelf4.fit(x_train, Y_train, batch_size=128, epochs=20, validation_split=0.2)
#%%
modelf5 = Sequential()
modelf5.add(Dense(4000, activation='relu', input_shape=(numpy.prod(X_test.shape[1:]),)))
modelf5.add(Dropout(0.3))
modelf5.add(Dense(2000, activation='relu'))
modelf5.add(Dropout(0.3))
modelf5.add(Dense(1000, activation='relu'))
modelf5.add(Dense(500, activation='relu'))
modelf5.add(Dropout(0.3))
modelf5.add(Dense(250, activation='relu'))
modelf5.add(Dense(100, activation='relu'))
modelf5.add(Dropout(0.3))
modelf5.add(Dense(10, activation='softmax'))
modelf5.summary()
modelf5.compile(optimizer=SGD(lr=0.001,decay=1e-7, momentum=0.9, nesterov=True), loss=categorical_crossentropy, metrics=['accuracy'])
#%%
modelf5.fit(x_train, Y_train, batch_size=128, epochs=20, validation_split=0.2)
#%%
test_loss, test_acc = modelf1.evaluate(X_test, Y_test)
print('test loss = ', test_loss, 'test accuracy = ', test_acc)
test_loss, test_acc = modelf2.evaluate(X_test, Y_test)
print('test loss = ', test_loss, 'test accuracy = ', test_acc)
test_loss, test_acc = modelf3.evaluate(X_test, Y_test)
print('test loss = ', test_loss, 'test accuracy = ', test_acc)
test_loss, test_acc = modelf4.evaluate(X_test, Y_test)
print('test loss = ', test_loss, 'test accuracy = ', test_acc)
test_loss, test_acc = modelf5.evaluate(X_test, Y_test)
print('test loss = ', test_loss, 'test accuracy = ', test_acc)
#%%
best_modelf = Sequential()
best_modelf.add(Dense(4000, activation='relu', input_shape=(numpy.prod(X_test.shape[1:]),)))
best_modelf.add(Dropout(0.3))
best_modelf.add(Dense(2000, activation='relu'))
best_modelf.add(Dropout(0.3))
best_modelf.add(Dense(1000, activation='relu'))
best_modelf.add(Dropout(0.3))
best_modelf.add(Dense(10, activation='softmax'))
best_modelf.summary()
best_modelf.compile(optimizer=SGD(lr=0.001,decay=1e-7, momentum=0.9, nesterov=True), loss=categorical_crossentropy, metrics=['accuracy'])
#%%
best_model.fit(x_train, Y_train, batch_size=128, epochs=20)
#%%
filters, biases = best_model.layers[1].get_weights()
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
n_filters, ix = 12, 1
figure(figsize=(20,10))
for i in range(n_filters):
	f = filters[:, :, :, i]
	ax = pyplot.subplot(n_filters, 3, ix)
	ax.set_xticks([])
	ax.set_yticks([])
	pyplot.imshow(cv2.cvtColor(f[:, :, :], cv2.COLOR_BGR2RGB))
	ix += 1
pyplot.show()
#%%
filters, biases = best_model.layers[1].get_weights()
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
n_filters, ix = 4, 1
for i in range(n_filters):
	f = filters[:, :, :, i]
	for j in range(3):
		ax = pyplot.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		pyplot.imshow(f[:, :, j], cmap='gray')
		ix += 1
pyplot.show()
#%%
ixs = [1, 3, 7, 9, 13]
outputs = [best_model.layers[i].output for i in ixs]
model = Model(inputs=best_model.inputs, outputs=outputs)
img = X_test[10,:,:,:]
img = expand_dims(img, axis=0)
feature_maps = model.predict(img)
for fm in feature_maps:
	ix = 1
	for _ in range(4):
		for _ in range(8):
			ax = pyplot.subplot(4, 8, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			pyplot.imshow(fm[0, :, :, ix-1], cmap='gray')
			ix += 1
	pyplot.show()
#%%
test = cv2.imread('1.png')
test = test.astype('float32')
test/= 255
counter = 0
pyramid = tuple(pyramid_gaussian(test, max_layer=-1, downscale=2, sigma=None, order=1, mode='reflect', cval=0, multichannel=True))
i = 0
while (pyramid[i].shape[0] > 32) & (pyramid[i].shape[1] > 32):
    image = pyramid[i]
    x_length = image.shape[0] - 32
    y_length = image.shape[1] - 32
    for j in range(0, x_length, int(x_length/80) + 1):
        for k in range(0, y_length, int(y_length/80) + 1):
            window = image[j:j + 32, k:k + 32,:]
            window = expand_dims(window, axis=0)
            y_predicted = best_model.predict(window)
            predicted_number = numpy.argmax(y_predicted, axis=1)
            score_of_predict = y_predicted[0,predicted_number]
            if score_of_predict > 0.9995:
              if counter == 0:
                final_matrix = numpy.array([i,j*2**i,k*2**i,score_of_predict[0],predicted_number[0]])
                counter = counter + 1
              else:
                final_matrix = numpy.vstack((final_matrix, [i,j*2**i,k*2**i,score_of_predict[0],predicted_number[0]]))
    i = i+1
number = final_matrix.shape[0]
final_size = 0
temp = 0
selected_predicted = final_matrix
while number != 0:
    temp_while = 0
    if number == 1:
        temp_max = 0
        if final_size == 0:
            final = numpy.array(selected_predicted)
            final_size = 1
        else:
            final = numpy.vstack((final, selected_predicted))
            final_size = final_size + 1
        break
    else:
        temp_max = numpy.where(selected_predicted[:, 3] == selected_predicted[:, 3].max())[0][0]
    if final_size == 0:
        final = numpy.array(selected_predicted[temp_max])
        final_size = 1
    else:
        final = numpy.vstack((final, selected_predicted[temp_max]))
        final_size = final_size + 1
    number = 0
    for i in range(0, selected_predicted.shape[0]):
        if (numpy.abs(selected_predicted[temp_max][1] - selected_predicted[i][1]) > 12) & (numpy.abs(selected_predicted[temp_max][2] - selected_predicted[i][2]) > 12):
            if temp_while == 0:
                filtered = numpy.array(selected_predicted[i])
                temp_while = 1
            else:
                filtered = numpy.vstack((filtered, selected_predicted[i]))
            number = number + 1
    if temp_while != 0:
        selected_predicted = filtered
print(final)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0,0,0)
lineType = 1
image =  pyramid[0]
final_temp = final
final = final.astype(int)
if final_size == 1:
    if final[4] == 1:
      if final[3] > 0.9998:
        image = cv2.rectangle(image, (final[2]*2**final[0], final[1]*2**final[0]), ((final[2] + 32)*2**final[0] , (final[1] + 32)*2**final[0] ), 0, 1)
        image = cv2.putText(image,str(final[4]), (final[2],final[1]), font, fontScale,fontColor,lineType)
    else:
        image = cv2.rectangle(image, (final[2]*2**final[0], final[1]*2**final[0]), ((final[2] + 32)*2**final[0] , (final[1] + 32)*2**final[0] ), 0, 1)
        image = cv2.putText(image,str(final[4]), (final[2],final[1]), font, fontScale,fontColor,lineType)

elif final_size > 1:
    for i in range(final_size):
       temp = final[i]
       temp2 = final_temp[i]
       if temp[4] == 1  :
         if temp2[3] > 0.9994:
              image = cv2.rectangle(image, (temp[2], temp[1]), (temp[2] + 32 * 2 ** temp[0], temp[1] + 32 * 2 ** temp[0]), 0, 1)
              image = cv2.putText(image,str(temp[4]), (temp[2],temp[1]), font, fontScale,fontColor,lineType)
       else:
               image = cv2.rectangle(image, (temp[2], temp[1]), (temp[2] + 32 * 2 ** temp[0], temp[1] + 32 * 2 ** temp[0]), 0, 1)
               image = cv2.putText(image,str(temp[4]), (temp[2],temp[1]), font, fontScale,fontColor,lineType)
image = 255 * image / image.max()
image = image.astype(numpy.uint8)
pyplot.imshow(image)
#%%
test = cv2.imread('1.png')
test = test.astype('float32')
test/= 255
counter = 0
pyramid = tuple(pyramid_gaussian(test, max_layer=-1, downscale=2, sigma=None, order=1, mode='reflect', cval=0, multichannel=True))
i = 0
while (pyramid[i].shape[0] > 32) & (pyramid[i].shape[1] > 32):
    image = pyramid[i]
    x_length = image.shape[0] - 32
    y_length = image.shape[1] - 32
    for j in range(0, x_length, int(x_length/80) + 1):
        for k in range(0, y_length, int(y_length/80) + 1):
            window = image[j:j + 32, k:k + 32,:]
            window = expand_dims(window, axis=0)
            window = window.reshape((len(window), numpy.prod(window.shape[1:])))
            y_predicted = best_modelf.predict(window)
            predicted_number = numpy.argmax(y_predicted, axis=1)
            score_of_predict = y_predicted[0,predicted_number]
            if score_of_predict > 0.6:
              if counter == 0:
                final_matrix = numpy.array([i,j*2**i,k*2**i,score_of_predict[0],predicted_number[0]])
                counter = counter + 1
              else:
                final_matrix = numpy.vstack((final_matrix, [i,j*2**i,k*2**i,score_of_predict[0],predicted_number[0]]))
    i = i+1
number = final_matrix.shape[0]
final_size = 0
temp = 0
selected_predicted = final_matrix
while number != 0:
    temp_while = 0
    if number == 1:
        temp_max = 0
        if final_size == 0:
            final = numpy.array(selected_predicted)
            final_size = 1
        else:
            final = numpy.vstack((final, selected_predicted))
            final_size = final_size + 1
        break
    else:
        temp_max = numpy.where(selected_predicted[:, 3] == selected_predicted[:, 3].max())[0][0]
    if final_size == 0:
        final = numpy.array(selected_predicted[temp_max])
        final_size = 1
    else:
        final = numpy.vstack((final, selected_predicted[temp_max]))
        final_size = final_size + 1
    number = 0
    for i in range(0, selected_predicted.shape[0]):
        if (numpy.abs(selected_predicted[temp_max][1] - selected_predicted[i][1]) > 8) & (numpy.abs(selected_predicted[temp_max][2] - selected_predicted[i][2]) > 8):
            if temp_while == 0:
                filtered = numpy.array(selected_predicted[i])
                temp_while = 1
            else:
                filtered = numpy.vstack((filtered, selected_predicted[i]))
            number = number + 1
    if temp_while != 0:
        selected_predicted = filtered
print(final)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0,0,0)
lineType = 1
image =  pyramid[0]
final_temp = final
final = final.astype(int)
if final_size == 1:
    if final[4] == 1:
      if final[3] > 0.9998:
        image = cv2.rectangle(image, (final[2]*2**final[0], final[1]*2**final[0]), ((final[2] + 32)*2**final[0] , (final[1] + 32)*2**final[0] ), 0, 1)
        image = cv2.putText(image,str(final[4]), (final[2],final[1]), font, fontScale,fontColor,lineType)

    else:
        image = cv2.rectangle(image, (final[2]*2**final[0], final[1]*2**final[0]), ((final[2] + 32)*2**final[0] , (final[1] + 32)*2**final[0] ), 0, 1)
        image = cv2.putText(image,str(final[4]), (final[2],final[1]), font, fontScale,fontColor,lineType)

elif final_size > 1:
    for i in range(final_size):
       temp = final[i]
       temp2 = final_temp[i]
       if temp[4] == 1  :
         if temp2[3] > 0.9994:
            image = cv2.rectangle(image, (temp[2], temp[1]), (temp[2] + 32 * 2 ** temp[0], temp[1] + 32 * 2 ** temp[0]), 0, 1)
            image = cv2.putText(image,str(temp[4]), (temp[2],temp[1]), font, fontScale,fontColor,lineType)
       else:
            image = cv2.rectangle(image, (temp[2], temp[1]), (temp[2] + 32 * 2 ** temp[0], temp[1] + 32 * 2 ** temp[0]), 0, 1)
            image = cv2.putText(image,str(temp[4]), (temp[2],temp[1]), font, fontScale,fontColor,lineType)
image = 255 * image / image.max()
image = image.astype(numpy.uint8)
cv2.imshow('image', image)