import cv2
import numpy
from sklearn.datasets import fetch_lfw_people
from skimage.transform import pyramid_gaussian
import pickle
from skimage import feature

faces = fetch_lfw_people()
positive_patches = faces.images
with open("classifier.file", "rb") as f:
    classifier = pickle.load(f)
image = cv2.imread('Test_images/test_image1.jpg', 0)
pyramid = tuple(pyramid_gaussian(image, max_layer=-1, downscale=2, sigma=None, order=1, mode='reflect', cval=0, multichannel=False))
temp = 0
i = 0
while (pyramid[i].shape[0] > positive_patches[0].shape[0]) & (pyramid[i].shape[1] > positive_patches[0].shape[1]):
    image = pyramid[i]
    x_length = image.shape[0] - positive_patches[0].shape[0]
    y_length = image.shape[1] - positive_patches[0].shape[1]
    for j in range(0, x_length, int(x_length/80) + 1):
        for k in range(0, y_length, int(y_length/80) + 1):
            window = image[j:j + positive_patches[0].shape[0], k:k + positive_patches[0].shape[1]]
            window_feature = feature.hog(window).reshape(1, -1)
            Y_predicted = classifier.predict(window_feature)
            if Y_predicted == 1:
                score = classifier.decision_function(window_feature)[0]
                if temp == 0:
                    all_predicted = numpy.array([i, j, k, score])
                    maximum = numpy.array([i, j, k, score])
                    temp = 1
                else:
                    all_predicted = numpy.vstack((all_predicted, [i, j, k, score]))
                    if score > maximum[3]:
                        maximum = numpy.array([i, j, k, score])
    i = i + 1
number = 0
temp = 0
image = pyramid[0]
for i in range(0, all_predicted.shape[0]):
    if all_predicted[i][3] > 0.9*maximum[3]:
        temp_predicted = all_predicted[i]
        if temp == 0:
            selected_predicted = numpy.array([temp_predicted[0], temp_predicted[1] * 2 ** temp_predicted[0], temp_predicted[2] * 2 ** temp_predicted[0], temp_predicted[3]])
            temp = 1
        else:
            selected_predicted = numpy.vstack((selected_predicted, [temp_predicted[0], temp_predicted[1] * 2 ** temp_predicted[0], temp_predicted[2] * 2 ** temp_predicted[0], temp_predicted[3]]))
        number = number + 1
final_size = 0
temp = 0
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
        print(numpy.abs(selected_predicted[temp_max][1] - selected_predicted[i][1]))
        if (numpy.abs(selected_predicted[temp_max][1] - selected_predicted[i][1]) > 100) & (numpy.abs(selected_predicted[temp_max][2] - selected_predicted[i][2]) > 100):
            if temp_while == 0:
                filtered = numpy.array(selected_predicted[i])
                temp_while = 1
            else:
                filtered = numpy.vstack((filtered, selected_predicted[i]))
            number = number + 1
    if temp_while != 0:
        selected_predicted = filtered
print(final)
final = final.astype(int)
if final_size == 1:
    image = cv2.rectangle(image, (final[2], final[1]), (final[2] + 47 * 2 ** final[0], final[1] + 62 * 2 ** final[0]), 0, 1)
elif final_size > 1:
    for i in range(final_size):
        temp = final[i]
        image = cv2.rectangle(image, (temp[2], temp[1]), (temp[2] + 47 * 2 ** temp[0], temp[1] + 62 * 2 ** temp[0]), 0, 1)
image = 255 * image / image.max()
image = image.astype(numpy.uint8)
cv2.imshow('image', image)
cv2.imwrite('Test_images/111_1.jpg', image)
cv2.waitKey(0)
cv2.HOGDescriptor()