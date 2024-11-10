import numpy as np
import cv2
import imutils
bg = None
aWeight = 0.5

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("asl_abc_2_crop_2_black.mp4 ")
top, right, bottom, left = 225,350,449,574
num_frames = 0

fgbg = cv2.createBackgroundSubtractorMOG2()
#fgbg = cv2.createBackgroundSubtractorMOG()
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=100):
    global bg
    cv2.imshow("image",image)
    fgim = fgbg.apply(image)
    cv2.imshow("fgimage", fgim)
    masked = cv2.bitwise_and(fgim,image,mask=fgim)
    cv2.imshow("Masked",masked)
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1] #25
    cv2.imshow("Thresholded",thresholded)
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

while(1):
    (ret,frame) = cap.read()
    #frame = cv2.imread("Gesture Image Data/D/1.jpg")
    cv2.imshow("original", frame)

    frame = cv2.flip(frame, 1)
    clone = frame.copy()
    (h,w) = frame.shape[:2]
    roi = frame[top:bottom, right:left]
    frame2 = roi.copy()


    converted = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    #skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    #cv2.imshow("HSV", skinMask)




    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    blur = cv2.GaussianBlur(gray,(7,7),0)
    #cv2.imshow("fg",fgmask)


    if num_frames<30:
        #a = 2
        run_avg(gray, aWeight)
    else:
        hand = segment(gray)
        if hand is not None:
            (thresholded, segmented) = hand
            cv2.drawContours(clone, [segmented+(right,top)], -1, (0,0,255))
            #thresholded = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)[1]
            cv2.imshow("Thresholded", thresholded)
    cv2.rectangle(clone, (left,top),(right,bottom),(0,255,0),2)
    num_frames += 1
    cv2.imshow("original", clone)

    #ret,thresh1 = cv2.threshold(blur,25,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #thresh2 = cv2.bitwise_not(thresh1)



    #cv2.imshow("Coins", thresh2)
    #fgmask = fgbg.apply(frame)
    #cv2.imshow("mask", fgmask)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
