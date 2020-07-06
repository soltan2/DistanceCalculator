import argparse
import time
import cv2
import numpy as np
import math

milisecondDelay = 0

#tracker = cv2.TrackerMOSSE_create()
# Can track
tracker = cv2.TrackerCSRT_create()
#tracker = cv2.TrackerMIL_create()
#tracker = cv2.TrackerMedianFlow_create()

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None then its from the webcam
if args.get("video", None) is None:
    cap = cv2.VideoCapture(1)
    time.sleep(2.0)

else:
    cap = cv2.VideoCapture(args["video"])
    #print("success")

success, frame = cap.read()

'''
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# define range of gray color in HSV
# change it according to your need !
lower_gray = np.array([0,50,20], dtype=np.uint8)
upper_gray = np.array([5,255,255], dtype=np.uint8)

mask = cv2.inRange(hsv, lower_gray, upper_gray)
# Threshold the HSV image to get only gray colors
frame = cv2.inRange(hsv, lower_gray, upper_gray)
# Bitwise-AND mask and original image
#frame = cv2.bitwise_and(frame,frame, mask= mask)'''

bbox = cv2.selectROI("Tracking",frame, False)
tracker.init(frame, bbox)

start = 0
end = 0
prev = [0,0,0,0]
curr = [1,0,0,0]
i = 0
startTime = 0
endTime = 0
finished = False
def drawBox(img,bbox, i):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 3 )
    global prev
    global curr
    global start
    global end
    global endTime
    global startTime
    global finished
    #print(x, y, x+w, y+h)
    if (i == 0):
        prev = [x,y,x+w,y+h]
        curr = [x,y,x+w,y+h]
        #print("Previous:", prev)
        #print("Current:", curr)
    else:
        prev = curr
        curr = [x,y,x+w,y+h]
        #print("Previous:", prev)
        #print("Current:", curr)
        #print(prev[1])
        deltaX = curr[0] - prev[0]
        deltaY = curr[1] - prev[1]
        if (deltaX > 0 and deltaY < 0 and start == 0):
            startTime = time.time()
            start = curr
            print("Initial Position", start, " ", startTime)
        if (startTime > 0):
            endTime = time.time() - startTime

        if (endTime >= 1 and finished is False):
            end = curr
            print("Ending Position", end, " ", endTime)
            finished = True
    cv2.putText(img, "Tracking", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

while(1):
    success, img = cap.read()

    '''
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of gray color in HSV
    # change it according to your need !
    lower_gray = np.array([0,50,20], dtype=np.uint8)
    upper_gray = np.array([5,255,255], dtype=np.uint8)

    # Threshold the HSV image to get only gray colors
    mask = cv2.inRange(hsv, lower_gray, upper_gray)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask = mask)
    #img = res
    img = mask'''

    success, bbox = tracker.update(img)
    if success:
        drawBox(img,bbox, i)
    else:
        cv2.putText(img, "Lost", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.rectangle(img,(15,15),(200,90),(255,0,255),2)
    cv2.putText(img, "Status:", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2);
    try:
        cv2.imshow("Tracking", img)
    except Exception as e:
        break
    ##cv2.imshow('frame',img)
    #cv2.imshow('mask',mask)
    #cv2.imshow('res',res)

    milisecondDelay = 1
    k = cv2.waitKey(milisecondDelay) & 0xFF
    if k == 27:
        break

    i += 1;


cv2.destroyAllWindows()
# Hypotenuse
pixelDistance = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
# Adjacent
pixelBottom = end[0] - start[0]
# Angle with inverse cosine (adjacent/hypotenuse)
# cos^-1(cos(theta)) = cos^-1(adjacent/hypotenuse)
# theta = cos^-1(adj/hypotenuse)
theta = math.acos(pixelBottom/pixelDistance)
print("Angle launched: ", theta)
pixelVelocity = pixelDistance/endTime
print("Velocity: ", pixelVelocity, "pixel/s")
y0 = start[1] - end[1]
#print(y0)

# Given formula
estimatedDistance = (pixelVelocity*math.cos(theta))/9.8 * (pixelVelocity*math.sin(theta) + math.sqrt((pixelVelocity*math.sin(theta))**2 + 2*9.8*y0))
print("Estimated Distance: ", estimatedDistance, " pixels")











'''ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if (cv2.contourArea(contour) > 1000 or cv2.contourArea(contour) < 800):
            continue

        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()'''
