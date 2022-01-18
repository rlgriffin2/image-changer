import cv2
import numpy as np
import imutils


def get_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 3)
    # thresh = cv2.threshold(blur, 150, 200, cv2.THRESH_BINARY)[1]
    # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 11)
    # thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_OTSU)[1]
    canny = cv2.Canny(blur.copy(), cv2.getTrackbarPos("Canny 1", "Trackbars"), cv2.getTrackbarPos("Canny 2", "Trackbars"))
    kernel = np.ones((5, 5))
    dilation = cv2.dilate(canny.copy(), kernel, iterations=1)
    cnts, hier = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cnts = imutils.grab_contours(cnts)
    return gray, blur, canny, dilation, cnts


def get_bounding_box(bounding_img, contours):
    global ref_obj
    # calibrate
    if ref_obj is None:
        # get largest contour area, should be post-it
        ref_obj = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(ref_obj)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        print(box)
        cv2.drawContours(bounding_img, [box], -1, (255, 255, 0), 3)
    else:
        for c in contours:
            if cv2.contourArea(c) > cv2.getTrackbarPos("Shape Area", "Trackbars"):
                cv2.drawContours(bounding_img, c, -1, (255, 255, 0), 3)
                perimeter = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02*perimeter, True)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(bounding_img, (x, y), (x+w, y+h), (0, 255, 255), 3)


def trackbars_init():
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 640, 240)
    # empty functions for lambdas because we don't need callback
    # intialize to values useful for reference, and then we can adjust it in the code below before start video
    cv2.createTrackbar("Canny 1", "Trackbars", 10, 255, lambda a: a)
    cv2.createTrackbar("Canny 2", "Trackbars", 30, 255, lambda a: a)
    cv2.createTrackbar("Shape Area", "Trackbars", 1500, 10000, lambda a: a)


# taken from https://www.youtube.com/watch?v=Fchzk1lDt7Q
def stack_images(scale, img_list):
    rows = len(img_list)
    cols = len(img_list[0])
    rows_available = isinstance(img_list[0], list)
    width = img_list[0][0].shape[1]
    height = img_list[0][0].shape[1]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_list[x][y].shape[:2] == img_list[0][0].shape[:2]:
                    img_list[x][y] = cv2.resize(img_list[x][y], (0, 0), None, scale, scale)
                else:
                    img_list[x][y] = cv2.resize(img_list[x][y], (img_list[0][0].shape[1], img_list[0][0].shape[0]), None, scale, scale)
                if len(img_list[x][y].shape) == 2: img_list[x][y] = cv2.cvtColor(img_list[x][y], cv2.COLOR_GRAY2BGR)
        img_blank = np.zeros((height, width, 3), np.uint8)
        hor = [img_blank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_list[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_list[x].shape[:2] == img_list[0].shape[:2]:
                img_list[x] = cv2.resize(img_list[x], (0, 0), None, scale, scale)
            else:
                img_list[x] = cv2.resize(img_list[x], (img_list[0].shape[1], img_list[0].shape[0]), None, scale, scale)
            if len(img_list[x].shape) == 2: img_list[x] = cv2.cvtColor(img_list[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_list)
        ver = hor
    return ver


trackbars_init()

# for reference/calibration
ref_obj = 'NOT NONE'
'''
known_width = 3.0
known_distance = 40.0
ref_img = cv2.imread("images/ref2.jpg")
copy = ref_img.copy()
ref_gray, ref_blur, ref_canny, ref_dilation, ref_cnts = get_contours(copy)
get_bounding_box(copy, ref_cnts)
cv2.imshow("ref", cv2.drawContours(copy, ref_obj, -1, (0, 0, 255), 3))
cv2.waitKey(0)'''

# distance from video = 44 in 
# for video
cv2.setTrackbarPos("Canny 1", "Trackbars", 115)
cv2.setTrackbarPos("Canny 2", "Trackbars", 158)
cv2.setTrackbarPos("Shape Area", "Trackbars", 1800)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray_img, blur_img, canny_img, dilation_img, cnts = get_contours(frame)
    bounding_img = frame.copy()
    get_bounding_box(bounding_img, cnts)
    cntr_img = cv2.drawContours(frame.copy(), cnts, -1, (255, 50, 50), 3)
    img_stack = stack_images(0.8, ([frame, blur_img, canny_img],
                                   [dilation_img, cntr_img, bounding_img]))
    cv2.imshow('frame', img_stack)
    char = cv2.waitKey(1)
    if char == 113:  # press q to quit
        break
cap.release()
cv2.destroyAllWindows()
'''
# for picture
image = cv2.imread('images/both.jpg')
cv2.imshow('test', get_bounding_box_moments(image, get_contours(image)))
cv2.waitKey(0)
'''
