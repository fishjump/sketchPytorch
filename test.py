import sketch_detector
import cv2

img = cv2.imread("test_case.jpg")
result = sketch_detector.get_sketch(img)
cv2.imshow("sketch_pytorch",result)
cv2.waitKey()