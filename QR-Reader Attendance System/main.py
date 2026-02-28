import os
import datetime
import time

import cv2
from pyzbar.pyzbar import decode
import matplotlib.pyplot as plt
import numpy as np

capture = cv2.VideoCapture(0)

while True:
    success, frame = capture.read()

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()