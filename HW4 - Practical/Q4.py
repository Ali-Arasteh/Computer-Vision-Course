import cv2
import numpy


def trim(img):
    if not numpy.sum(img[0]):
        return trim(img[1:])
    if not numpy.sum(img[-1]):
        return trim(img[:-2])
    if not numpy.sum(img[:, 0]):
        return trim(img[:, 1:])
    if not numpy.sum(img[:, -1]):
        return trim(img[:, :-2])
    return img


image1 = cv2.imread('images/3-1.jpeg')
image1 = cv2.resize(image1, (int(image1.shape[1] * 0.6), int(image1.shape[0] * 0.6)), interpolation=cv2.INTER_AREA)
image2 = cv2.imread('images/3-2.jpeg')
image2 = cv2.resize(image2, (int(image2.shape[1] * 0.6), int(image2.shape[0] * 0.6)), interpolation=cv2.INTER_AREA)

akaze = cv2.AKAZE_create()
keyPoints1, desc1 = akaze.detectAndCompute(image1, None)
keyPoints2, desc2 = akaze.detectAndCompute(image2, None)
cv2.imshow('3_1', cv2.drawKeypoints(image1, keyPoints1, None))
cv2.imwrite('images/3-1 interest points.jpeg', cv2.drawKeypoints(image1, keyPoints1, None))
cv2.imshow('3_2', cv2.drawKeypoints(image2, keyPoints2, None))
cv2.imwrite('images/3-2 interest points.jpeg', cv2.drawKeypoints(image2, keyPoints2, None))
cv2.waitKey(0)
cv2.destroyAllWindows()
matchedPoints = cv2.BFMatcher().knnMatch(desc1, desc2, k=2)
selected = []
for m, n in matchedPoints:
    if m.distance < 0.5 * n.distance:
        selected.append(m)
drawParams = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)
image = cv2.drawMatches(image1, keyPoints1, image2, keyPoints2, selected, None, **drawParams)
cv2.imshow('matched points', image)
cv2.imwrite('images/matched points.jpeg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
srcPoints = numpy.float32([keyPoints2[m.trainIdx].pt for m in selected]).reshape(-1, 1, 2)
dstPoints = numpy.float32([keyPoints1[m.queryIdx].pt for m in selected]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
dst = cv2.warpPerspective(image2, M, (image1.shape[1] + image2.shape[1], image1.shape[0]))
dst[0:image1.shape[0], 0:image1.shape[1]] = image1
cv2.imshow('trimmed perspective image', trim(dst))
cv2.imwrite('images/trimmed perspective image.jpeg', trim(dst))
cv2.waitKey(0)
cv2.destroyAllWindows()
