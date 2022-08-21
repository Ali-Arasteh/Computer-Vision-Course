import cv2


videoCapture = cv2.VideoCapture(0)
videoWriter = cv2.VideoWriter('video/videoWriter.avi', cv2.VideoWriter_fourcc(*"MJPG"), 50.0, (1000, 1000))
save = 0
while videoCapture.isOpened():
    ret, frame = videoCapture.read()
    cv2.imshow('Camera', frame)
    if save == 1:
        videoWriter.write(frame)
    inputKey = cv2.waitKey(1)
    if inputKey == ord('s'):
        save = 1
    elif inputKey == ord('e'):
        if save == 1:
            save = 0
        else:
            break
videoCapture.release()
videoWriter.release()
cv2.destroyAllWindows()
