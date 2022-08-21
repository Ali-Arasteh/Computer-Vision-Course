import cv2


image = cv2.imread('image/3.jpg', cv2.IMREAD_GRAYSCALE)
params = cv2.SimpleBlobDetector_Params()
params.minDistBetweenBlobs = 25
params.filterByArea = True
params.minArea = 50
params.maxArea = 5000
params.minThreshold = 50
params.maxThreshold = 240
params.thresholdStep = 20
params.filterByCircularity = True
params.minCircularity = 0.5
params.maxCircularity = 1
params.filterByInertia = True
params.minInertiaRatio = 0.05
params.maxInertiaRatio = 1
params.filterByConvexity = True
params.minConvexity = 0.8
params.maxConvexity = 1
detector = cv2.SimpleBlobDetector_create(params)
keyPoints = detector.detect(image)
modifiedImage = cv2.drawKeypoints(image, keyPoints, None, color=(0, 0, 255), flags=0)
cv2.imshow('Blob detection', modifiedImage)
cv2.imwrite('image/Blob detection.jpg', modifiedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
