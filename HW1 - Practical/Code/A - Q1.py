import cv2


# A - Q1 - part A
image = cv2.imread('image/1.jpg')
cv2.putText(image, '96101165', (0, 25), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 0, 0), 5)
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Labeling RGB Image', image)
cv2.imshow('Labeling Grayscale image', grayImage)
while True:
    inputKey = cv2.waitKey(0)
    if inputKey == ord('s') or inputKey == ord('e'):
        break
if inputKey == ord('s'):
    cv2.imwrite('image/Labeling RGB Image.jpg', image)
    cv2.imwrite('image/Labeling Grayscale Image.jpg', grayImage)
    cv2.destroyAllWindows()
elif inputKey == ord('e'):
    cv2.destroyAllWindows()

# A - Q1 - part B
image = cv2.imread('image/football.jpg')
cv2.rectangle(image, (300, 460), (375, 530), (0, 0, 0), 1)
cv2.imshow('Ball Recognizing Image', image)
cv2.imwrite('image/Ball Recognizing Image.jpg', image)
image = cv2.imread('image/football.jpg')
image[465:530, 630:700] = image[460:525, 300:370]
cv2.imshow('Ball Coping', image)
cv2.imwrite('image/Ball Coping.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
