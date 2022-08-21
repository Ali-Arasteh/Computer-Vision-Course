import cv2
import numpy

videoCapture = cv2.VideoCapture('video/video.mp4')
videoCapture.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
Frames = []
while videoCapture.isOpened():
    ret, frame = videoCapture.read()
    if not ret:
        break
    Frames.append(frame)
videoCapture.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
background = numpy.median(Frames, axis=0).astype(dtype=numpy.uint8)
cv2.imshow('Background', background)
cv2.imwrite('image/Background.jpg', background)
cv2.waitKey(0)
grayBackground = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
while videoCapture.isOpened():
    ret, frame = videoCapture.read()
    if not ret:
        break
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    difference = cv2.subtract(grayFrame, grayBackground)
    cv2.imshow('Difference', difference)
    k = cv2.waitKey(30)
    if k == ord('e'):
        break
videoCapture.release()
cv2.destroyAllWindows()
