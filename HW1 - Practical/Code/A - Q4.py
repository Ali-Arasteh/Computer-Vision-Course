import cv2
import numpy


imageName = '2.jpg'
image = cv2.imread('image/' + imageName, cv2.IMREAD_GRAYSCALE)
if image.shape[0] > 1000:
    image = cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)), interpolation=cv2.INTER_AREA)


def sobel_on_change(value):
    _, thresholdSobelImage = cv2.threshold(sobelImage, value, 255, cv2.THRESH_BINARY)
    cv2.imshow('Sobel edge detection', thresholdSobelImage)
    cv2.imwrite('image/Sobel edge detection ' + imageName, thresholdSobelImage)



xSobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
ySobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobelImage = (xSobel ** 2 + ySobel ** 2) ** (1 / 2)
cv2.namedWindow('Sobel edge detection')
cv2.createTrackbar('Threshold', 'Sobel edge detection', 100, 255, sobel_on_change)
sobel_on_change(120)
cv2.waitKey(0)
cv2.destroyAllWindows()


def canny_on_change(value):
    minValue = cv2.getTrackbarPos('Min Value', 'Canny edge detection')
    maxValue = cv2.getTrackbarPos('Max Value', 'Canny edge detection')
    cannyImage = cv2.Canny(image, minValue, maxValue, True)
    cv2.imshow('Canny edge detection', cannyImage)
    cv2.imwrite('image/Canny edge detection ' + imageName, cannyImage)


cv2.namedWindow('Canny edge detection')
cv2.createTrackbar('Min Value', 'Canny edge detection', 50, 1440, canny_on_change)
cv2.createTrackbar('Max Value', 'Canny edge detection', 150, 1440, canny_on_change)
canny_on_change(0)
cv2.waitKey(0)
cv2.destroyAllWindows()


def log_on_change(value):
    _, thresholdLoGImage = cv2.threshold(LoGImage, value, 255, cv2.THRESH_BINARY)
    cv2.imshow('LoG edge detection', thresholdLoGImage)
    cv2.imwrite('image/LoG edge detection ' + imageName, thresholdLoGImage)


gaussianFilteredImage = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)
LoGImage = cv2.Laplacian(gaussianFilteredImage, cv2.CV_64F, ksize=3)
LoGImage = numpy.absolute(LoGImage)
cv2.namedWindow('LoG edge detection')
cv2.createTrackbar('Threshold', 'LoG edge detection', 50, 500, log_on_change)
log_on_change(50)
cv2.waitKey(0)
cv2.destroyAllWindows()
