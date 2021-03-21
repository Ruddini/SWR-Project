import cv2
import line_detection

vid = cv2.VideoCapture(0)

while (True):
    a = line_detection.LineDetector(vid)
    e = a.detection_color_line()
    cv2.imshow('frame3', e)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

