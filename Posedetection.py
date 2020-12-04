import cv2
import time
import numpy as np
MODE = "COCO"

TIMER = int(5)


if MODE is "COCO":
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
elif MODE is "MPI" :
    protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

videoCaptureObject = cv2.VideoCapture(0)
result = True
def imgcapture(frame):
    image = cv2.imwrite("NewImage.jpg", frame)
    # return image

def processImage(frame):
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    t = time.time()
    # input image dimensions for the network
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()

    print("time taken by network : {:.3f}".format(time.time() - t))

    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold:
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else:
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.imwrite('Output-Keypoints.jpg', frameCopy)
    cv2.imwrite('Output-Skeleton.jpg', frame)

def timerCount(frame):
    TIMER = int(5)
    prev = time.time()
    while TIMER >= 0:
        img = cv2.imread("timer.png")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(TIMER),
                    (100,100), font,
                    3, (255, 0, 0),
                    2, cv2.LINE_AA)
        cv2.imshow('a', img)
        cv2.waitKey(125)
        cur = time.time()

        if cur - prev >= 1:
            prev = cur
            TIMER = TIMER - 1
prev = time.time()
k = ""
while True:
    ret,frame = videoCaptureObject.read()
    # cv2.imwrite("NewPicture.jpg",frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,
                'PRESS "c" TO CAPTURE IMAGE',
                (50, 50),
                font, 1,
                (0, 255, 255),
                2,
                cv2.LINE_4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, str(TIMER),
                (110, 110), font,
                2, (255, 0, 0),
                2, cv2.LINE_AA)
    # cur = time.time()
    # re = cur - prev
    # if cur - prev >= 1:
    #     prev = cur
    #     TIMER = TIMER - 1
    # if (re == 0):
    #     imgcapture(frame)
    #     k = 'q'


    cv2.imshow("Imput", frame)
    key = cv2.waitKey(1)
    cur = time.time()
    if cur - prev >= 1:
        prev = cur
        TIMER = TIMER - 1
    if (TIMER == -1):
        imgcapture(frame)
        key = 'q'
    if key == ord('q') or key == 'q':
        break


fr = cv2.imread("NewImage.jpg")
processImage(fr)
out1 = cv2.imread("Output-Keypoints.jpg")
out2 = cv2.imread("Output-Skeleton.jpg")
cv2.imshow("Output Key point", out1)
cv2.imshow("Output Skeleton", out2)
cv2.waitKey(0)
videoCaptureObject.release()
cv2.destroyAllWindows()