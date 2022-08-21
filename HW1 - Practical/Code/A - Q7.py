import cv2
import numpy

image = cv2.imread("image/4.jpg")
exactNoseImage = image.copy()
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
modifiedImage = cv2.threshold(grayImage, 100, 1, cv2.THRESH_BINARY_INV)[1]
noseSample = numpy.zeros((18, 32))
noseSample[:, :] = 0
noseSample[0, :] = 1
noseSample[-1, :] = 1
noseSample[:, 0] = 1
noseSample[:, -1] = 1
filteredImage = cv2.filter2D(modifiedImage, -1, noseSample)
modifiedFilteredImage = cv2.threshold(filteredImage, 80, 255, cv2.THRESH_BINARY)[1]
position = numpy.argwhere(modifiedFilteredImage == 255)
for i in range(position.shape[0]):
    image = cv2.circle(image, (position[i, 1], position[i, 0]), 25, (0, 0, 255))
cv2.imshow("Nose Recognition Image", image)
cv2.imwrite("image/Nose Recognition Image.jpg", image)
print(0.5 * image.shape[1])
for i in range(position.shape[0]):
    if 0.33 * image.shape[0] <= position[i, 0] <= 0.66 * image.shape[0] and position[i, 1] <= 0.5 * image.shape[1]:
        print(position[i, :])
        exactNoseImage = cv2.circle(exactNoseImage, (position[i, 1], position[i, 0]), 25, (0, 0, 255))
cv2.imshow("Exact Nose Recognition Image", exactNoseImage)
cv2.imwrite("image/Exact Nose Recognition Image.jpg", exactNoseImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
