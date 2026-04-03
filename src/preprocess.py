import cv2
import numpy as np

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def crop_conjunctiva(img):

    # =========================
    #EYE DETECTION
    # =========================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    if len(eyes) > 0:
        x, y, w_eye, h_eye = eyes[0]

        # Expand downward to include conjunctiva
        pad = int(0.4 * h_eye)
        y_new = max(y + h_eye//2 - pad, 0)
        h_new = min(h_eye + pad, img.shape[0] - y_new)

        eye_crop = img[y_new:y_new+h_new, x:x+w_eye]
    else:
        # Fallback if eye not detected
        eye_crop = img

    # =========================
    #HSV MASK ON EYE REGION
    # =========================
    hsv = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 70, 100])
    upper = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    red_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])

    if red_ratio < 0.01:  # very mild filter
        # don't reject, just reduce impact by using fallback
        contours = []

    # =========================
    #CLEAN MASK
    # =========================
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # =========================
    #CONTOUR CROP
    # =========================
    if contours:
        contours = [c for c in contours if cv2.contourArea(c) > 500]

        if contours:
            c = max(contours, key=cv2.contourArea)

            x, y, w_box, h_box = cv2.boundingRect(c)

            pad = 15
            x = max(x - pad, 0)
            y = max(y - pad, 0)

            h_e, w_e = eye_crop.shape[:2]

            w_box = min(w_box + 2*pad, w_e - x)
            h_box = min(h_box + 2*pad, h_e - y)

            cropped = eye_crop[y:y+h_box, x:x+w_box]
            return cv2.resize(cropped, (224, 224))

    # =========================
    # SMART FALLBACK (CENTER OF EYE REGION)
    # =========================
    h_e, w_e = eye_crop.shape[:2]

    cx, cy = w_e // 2, h_e // 2
    size = min(h_e, w_e) // 2

    x1 = max(cx - size//2, 0)
    y1 = max(cy - size//2, 0)
    x2 = x1 + size
    y2 = y1 + size

    cropped = eye_crop[y1:y2, x1:x2]
