import cv2
import numpy


videoCapture = cv2.VideoCapture('video/movie.avi')
videoCapture.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
while videoCapture.isOpened():
    ret, frame = videoCapture.read()
    if not ret:
        break
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cannyFrame = cv2.Canny(grayFrame, 20, 30)
    xSobel = cv2.Sobel(grayFrame, cv2.CV_64F, 1, 0, ksize=3)
    ySobel = cv2.Sobel(grayFrame, cv2.CV_64F, 0, 1, ksize=3)
    sobelFrame = (xSobel ** 2 + ySobel ** 2) ** (1 / 2)
    _, thresholdSobelFrame = cv2.threshold(sobelFrame, 25, 255, cv2.THRESH_BINARY)
    xKernel = numpy.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    yKernel = numpy.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    xPrewitt = cv2.filter2D(grayFrame, -1, xKernel)
    yPrewitt = cv2.filter2D(grayFrame, -1, yKernel)
    prewittFrame = (xPrewitt ** 2 + yPrewitt ** 2) ** (1 / 2)
    _, thresholdPrewittFrame = cv2.threshold(prewittFrame, 10, 255, cv2.THRESH_BINARY)
    cv2.imshow('Canny edge detection', cannyFrame)
    cv2.imshow('Sobel edge detection', thresholdSobelFrame)
    cv2.imshow('Prewitt edge detection', thresholdPrewittFrame)
    k = cv2.waitKey(30)
    if k == ord('e'):
        break
cv2.imwrite('image/Canny edge detection.jpg', cannyFrame)
cv2.imwrite('image/Sobel edge detection.jpg', thresholdSobelFrame)
cv2.imwrite('image/Prewitt edge detection.jpg', thresholdPrewittFrame)
videoCapture.release()
cv2.destroyAllWindows()


videoCapture = cv2.VideoCapture('video/movie.avi')
videoCapture.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
while videoCapture.isOpened():
    ret, frame = videoCapture.read()
    if not ret:
        break
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayFrame = cv2.GaussianBlur(grayFrame, (5, 5), cv2.BORDER_DEFAULT)
    cannyFrame = cv2.Canny(grayFrame, 20, 30)
    xSobel = cv2.Sobel(grayFrame, cv2.CV_64F, 1, 0, ksize=3)
    ySobel = cv2.Sobel(grayFrame, cv2.CV_64F, 0, 1, ksize=3)
    sobelFrame = (xSobel ** 2 + ySobel ** 2) ** (1 / 2)
    _, thresholdSobelFrame = cv2.threshold(sobelFrame, 25, 255, cv2.THRESH_BINARY)
    xKernel = numpy.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    yKernel = numpy.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    xPrewitt = cv2.filter2D(grayFrame, -1, xKernel)
    yPrewitt = cv2.filter2D(grayFrame, -1, yKernel)
    prewittFrame = (xPrewitt ** 2 + yPrewitt ** 2) ** (1 / 2)
    _, thresholdPrewittFrame = cv2.threshold(prewittFrame, 10, 255, cv2.THRESH_BINARY)
    cv2.imshow('Canny edge detection with gaussian filter', cannyFrame)
    cv2.imshow('Sobel edge detection with gaussian filter', thresholdSobelFrame)
    cv2.imshow('Prewitt edge detection with gaussian filter', thresholdPrewittFrame)
    k = cv2.waitKey(30)
    if k == ord('e'):
        break
cv2.imwrite('image/Canny edge detection with gaussian filter.jpg', cannyFrame)
cv2.imwrite('image/Sobel edge detection with gaussian filter.jpg', thresholdSobelFrame)
cv2.imwrite('image/Prewitt edge detection with gaussian filter.jpg', thresholdPrewittFrame)
videoCapture.release()
cv2.destroyAllWindows()
