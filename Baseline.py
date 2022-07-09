from itertools import count
import tempfile
import cv2
from cv2 import sqrt
import mediapipe as mp
import time
import sklearn
from sklearn.linear_model import LinearRegression
import numpy as np
import keras
import keyboard

PREDICT_TIME_FRAME = 3  # How long the algorithm will store past frames
PREDICT_TIME_THRESHOLD = 10  # How long the algorithm will predict after tracking lost
# LIMIT_HAND_SIZE = 0.01  # How big the delta with centroids and landmarks can get relative to window size

l_prev_HAND_queue = []  # Left hand frame queue
r_prev_HAND_queue = []  # Right hand frame queue

l_prev_CENX_queue = []
l_prev_CENY_queue = []
r_prev_CENX_queue = []
r_prev_CENY_queue = []

l_time_since_last_detection = 0  # Left hand time since last detection
r_time_since_last_detection = 0  # Right hand time since last detection

# Start video capture using webcam
_videoCapture = cv2.VideoCapture(0)
_videoCapture.set(3, 1920)
_videoCapture.set(4, 1080)

# Get mediapipline for hand detection baseline software
_mediaPipeHands = mp.solutions.hands
_handsDetection = _mediaPipeHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.3)
_mediaPipeDraw = mp.solutions.drawing_utils

# Variables for FPS / frame calculations
_pTime = 0
_cTime = 0

totalFrame = 0
notTrackFrame = 0
predictFrame = 0

UNETModel = keras.models.load_model("unet.h5")
trackBool = True
lastFramePress = False
didPredict =  False
didCalc = False

globPredX = []
globPredY = []
globCalcX = []
globCalcY = []

# Loop for tracking / main loop
while True:
    totalFrame += 1
    if keyboard.is_pressed("q"):
        if lastFramePress == False:
            trackBool = not trackBool
            lastFramePress = True
    else:
        lastFramePress = False
    
    success, img = _videoCapture.read()  # Get image from webcam
    rgbIMG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Change image from BGR to RGB
    hsvIMG = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    results = _handsDetection.process(rgbIMG)  # Get hands from mediapipeline

    h, s, v = cv2.split(hsvIMG)
    s = s * 1.5
    s = np.clip(s,0,255)
    hsvIMG = cv2.merge((h.astype(np.uint8),s.astype(np.uint8),v.astype(np.uint8)))

    hsvIMG = cv2.resize(hsvIMG, dsize=(128,128), interpolation=cv2.INTER_AREA)
    hsvIMG = hsvIMG.astype(float) / 255

    pred_img = UNETModel.predict(np.array(hsvIMG).reshape(-1, 128, 128, 3), verbose=0)
    pred_img = pred_img.reshape(128, 128)
    pred_img = cv2.resize(pred_img, dsize=(1920, 1080), interpolation=cv2.INTER_AREA)

    ret,thresh = cv2.threshold(np.abs(pred_img - 1),0.5,1,0)
    M = cv2.moments(thresh)
    
    try:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # print(cX, cY);
        cv2.circle(img, (cX, cY), 20, (1, 1, 1), cv2.FILLED)
    except:
        cX = 0
        cY = 0
    
    if results.multi_hand_landmarks:  # If program can detect hands
        # Add hand landmark data to hand queues
        for handLimbs, hand in zip(results.multi_hand_landmarks, results.multi_handedness):
            handType = hand.classification[0].label
            handType = "Left" if handType == "Right" else "Right"  # Labels are inverted in multi_handedness.classification[0].label

            # Check what hands currently are on screen
            if handType == "Left" or handType == "Right":
                l_prev_HAND_queue.append(handLimbs)
                l_prev_CENX_queue.append(float(cX / 1920.0))
                l_prev_CENY_queue.append(float(cY / 1080.0))
                if len(l_prev_HAND_queue) > PREDICT_TIME_FRAME:  # If the size of the queue is past the max time frame, remove the first element
                    l_prev_HAND_queue.pop(0)
                    l_prev_CENX_queue.pop(0)
                    l_prev_CENY_queue.pop(0)
                l_time_since_last_detection = 0

            '''
            if handType == "Right":
                r_prev_HAND_queue.append(handLimbs)
                r_prev_CENX_queue.append(float(cX / 1920.0))
                r_prev_CENY_queue.append(float(cY / 1080.0))
                if len(r_prev_HAND_queue) > PREDICT_TIME_FRAME:  # If the size of the queue is past the max time frame, remove the first element
                    r_prev_HAND_queue.pop(0)
                    r_prev_CENX_queue.pop(0)
                    r_prev_CENY_queue.pop(0)
                r_time_since_last_detection = 0
            '''

    # See what hands are currently on screen
    hands = []
    if results.multi_hand_landmarks:
        for hand in results.multi_handedness:
            hands.append("Left" if hand.classification[0].label == "Right" else "Right")   # Labels are inverted in multi_handedness.classification[0].label

    # Prediction when left hand doesn't exist in frame
    # if not "Left" in hands or not "Right" in hands:
    if not "Right" in hands and not "Left" in hands:
        notTrackFrame += 1
        # Only start prediction if there is enough data and time since last detection is not over the threshold
        if len(l_prev_HAND_queue) == PREDICT_TIME_FRAME and l_time_since_last_detection < PREDICT_TIME_THRESHOLD:
            predictFrame += 1
            didPredict = True
            x_data = [[] for i in range(21)]  # Holds x-cen values of all landmarks of all timeframes, 2D array: [21][TIMEFRAME]
            y_data = [[] for i in range(21)]  # Holds y-cen values of all landmarks of all timeframes, 2D array: [21][TIMEFRAME]
            centroid_x_data = []  # Holds centroid x value of all landmarks at a timeframe, 1D array: [TIMEFRAME]
            centroid_y_data = []  # Holds centroid y value of all landmarks at a timeframe, 1D array: [TIMEFRAME]

            '''
            # Get centroid positions for all data within timeframe
            # Centroids are calculated with a simple average of all landmarks
            for pasthand in l_prev_HAND_queue:  # Using l_prev_HAND_queue[i].landmark.limb.x for finding position of certain node at certain time
                xsum = 0
                ysum = 0
                for id, limb in enumerate(pasthand.landmark):
                    xsum += limb.x
                    ysum += limb.y
                centroid_x_data.append(xsum / 21.0)
                centroid_y_data.append(ysum / 21.0)
            '''

            # Get x, y positions of landmarks relative to centroid positions for all data within timeframe
            for pasthand, xcen, ycen in zip(l_prev_HAND_queue, l_prev_CENX_queue, l_prev_CENY_queue):  # Using l_prev_HAND_queue[i].landmark.limb.x for finding position of certain node at certain time
                for id, limb in enumerate(pasthand.landmark):
                    x_data[id].append(limb.x - xcen)
                    y_data[id].append(limb.y - ycen)

            pred_x = []  # 21 x values that will be predicted
            pred_y = []  # 21 y values that will be predicted
            diff_pred_x = []  # 21 x-cen values that will be predicted
            diff_pred_y = []  # 21 y-cen values that will be predicted
            pred_centroid_x = 0  # centroid x value that will be predicted
            pred_centroid_y = 0  # centroid y value that will be predicted

            '''
            model = LinearRegression()  # Use regression on centroid x, which are each stored in a 1 dimensional array
            model.fit(np.array([i for i in range(PREDICT_TIME_FRAME)]).reshape(-1, 1), np.array(centroid_x_data))
            pred_centroid_x = model.predict(np.array(l_time_since_last_detection + PREDICT_TIME_FRAME).reshape(1, -1))

            model = LinearRegression()  # Use regression on centroid y, which are each stored in a 1 dimensional array
            model.fit(np.array([i for i in range(PREDICT_TIME_FRAME)]).reshape(-1, 1), np.array(centroid_y_data))
            pred_centroid_y = model.predict(np.array(l_time_since_last_detection + PREDICT_TIME_FRAME).reshape(1, -1))
            '''

            for node in range(21):
                model = LinearRegression()  # Use linear regression on certain nodes, which are each stored in a 1 dimensional array
                model = sklearn.svm.SVR()
                model.fit(np.array([i for i in range(PREDICT_TIME_FRAME)]).reshape(-1, 1), np.array(x_data[node]))
                diff_pred_x.append(model.predict(np.array(l_time_since_last_detection + PREDICT_TIME_FRAME).reshape(1, -1)))

                model = LinearRegression()  # Use linear regression on certain nodes, which are each stored in a 1 dimensional array
                model = sklearn.svm.SVR()
                model.fit(np.array([i for i in range(PREDICT_TIME_FRAME)]).reshape(-1, 1), np.array(y_data[node]))
                diff_pred_y.append(model.predict(np.array(l_time_since_last_detection + PREDICT_TIME_FRAME).reshape(1, -1)))

            
            

            # print(diff_pred_x, diff_pred_y)
            # Calculate actual position of each node with data that is relative to centroids
            for node in range(21):
                pred_x.append(cX / 1920.0 + diff_pred_x[node])
                pred_y.append(cY / 1080.0 + diff_pred_y[node])

            # print(pred_x, pred_y)
            # print(pred_x)
            # print(pred_y)

            # Draw circles in predicted places
            for x, y, id in zip(pred_x, pred_y, [i for i in range(21)]):
                cv2.circle(img, (int(x * width), int(y * height)), 10, (id * int(255 / 21), id * int(255 / 21), id * int(255 / 21)), cv2.FILLED)
                globPredX.append(int(x * width))
                globPredY.append(int(y * height))
            connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13 ,14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
            for i in range(len(pred_x)):
                for j in range(len(pred_x)):
                    if (i, j) in connections:
                        cv2.line(img, (int(pred_x[i] * width), int(pred_y[i] * height)), (int(pred_x[j] * width), int(pred_y[j] * height)), (212, 175, 255), 3)

            # cv2.circle(img, (int(cX * width), int(cY * height)), 20, (255, 255, 255), cv2.FILLED)
        l_time_since_last_detection += 1

    '''
    # Prediction when left hand doesn't exist in frame
    if not "Right" in hands:
        # Only start prediction if there is enough data and time since last detection is not over the threshold
        if len(r_prev_HAND_queue) == PREDICT_TIME_FRAME and r_time_since_last_detection < PREDICT_TIME_THRESHOLD:
            x_data = [[] for i in range(21)]  # Holds x-cen values of all landmarks of all timeframes, 2D array [21][TIMEFRAME]
            y_data = [[] for i in range(21)]  # Holds y-cen values of all landmarks of all timeframes, 2D array [21][TIMEFRAME]
            centroid_x_data = []  # Holds centroid x value of all landmarks at a timeframe, 1D array [TIMEFRAME]
            centroid_y_data = []  # Holds centroid y value of all landmarks at a timeframe, 1D array [TIMEFRAME]
            
            
            # Get centroid positions for all data within timeframe
            # Centroids are calculated with a simple average of all landmarks
            for pasthand in r_prev_HAND_queue:  # Using l_prev_HAND_queue[i].landmark.limb.x for finding position of certain node at certain time
                xsum = 0
                ysum = 0
                for id, limb in enumerate(pasthand.landmark):
                    xsum += limb.x
                    ysum += limb.y
                centroid_x_data.append(xsum / 21.0)
                centroid_y_data.append(ysum / 21.0)
            

            # Get x, y positions of landmarks relative to centroid positions for all data within timeframe
            for pasthand, xcen, ycen in zip(r_prev_HAND_queue, r_prev_CENX_queue, r_prev_CENY_queue):  # Using l_prev_HAND_queue[i].landmark.limb.x for finding position of certain node at certain time
                for id, limb in enumerate(pasthand.landmark):
                    x_data[id].append(limb.x - xcen)
                    y_data[id].append(limb.y - ycen)

            pred_x = []  # 21 x values that will be predicted
            pred_y = []  # 21 y values that will be predicted
            diff_pred_x = []  # 21 x-cen values that will be predicted
            diff_pred_y = []  # 21 y-cen values that will be predicted
            pred_centroid_x = 0  # centroid x value that will be predicted
            pred_centroid_y = 0  # centroid y value that will be predicted
            
            
            model = LinearRegression()  # Use regression on centroid x, which are each stored in a 1 dimensional array
            model.fit(np.array([i for i in range(PREDICT_TIME_FRAME)]).reshape(-1, 1), np.array(centroid_x_data))
            pred_centroid_x = model.predict(np.array(r_time_since_last_detection + PREDICT_TIME_FRAME).reshape(1, -1))

            model = LinearRegression()  # Use regression on centroid y, which are each stored in a 1 dimensional array
            model.fit(np.array([i for i in range(PREDICT_TIME_FRAME)]).reshape(-1, 1), np.array(centroid_y_data))
            pred_centroid_y = model.predict(np.array(r_time_since_last_detection + PREDICT_TIME_FRAME).reshape(1, -1))
            

            for node in range(21):
                # model = LinearRegression()  # Use linear regression on certain nodes, which are each stored in a 1 dimensional array
                model = sklearn.svm.SVR()
                model.fit(np.array([i for i in range(PREDICT_TIME_FRAME)]).reshape(-1, 1), np.array(x_data[node]))
                diff_pred_x.append(model.predict(np.array(r_time_since_last_detection + PREDICT_TIME_FRAME).reshape(1, -1)))

                # model = LinearRegression()  # Use linear regression on certain nodes, which are each stored in a 1 dimensional array
                model = sklearn.svm.SVR()
                model.fit(np.array([i for i in range(PREDICT_TIME_FRAME)]).reshape(-1, 1), np.array(y_data[node]))
                diff_pred_y.append(model.predict(np.array(r_time_since_last_detection + PREDICT_TIME_FRAME).reshape(1, -1)))

            # Calculate actual position of each node with data that is relative to centroids
            for node in range(21):
                pred_x.append(cX / 1920.0 + diff_pred_x[node])
                pred_y.append(cY / 1080.0 + diff_pred_y[node])

            # print(pred_x)
            # print(pred_y)

            # Draw circles in predicted places
            for x, y, id in zip(pred_x, pred_y, [i for i in range(21)]):
                cv2.circle(img, (int(x * width), int(y * height)), 10, (id * int(255 / 21), id * int(255 / 21), id * int(255 / 21)), cv2.FILLED)
            # cv2.circle(img, (int(cX * width), int(cY * height)), 20, (255, 255, 255), cv2.FILLED)
        r_time_since_last_detection += 1
    '''
    # If hands are tracked
    sumx = 0
    sumy = 0
    if results.multi_hand_landmarks:
        didCalc = True
        for handLimbs in results.multi_hand_landmarks:  # Enumerate between hands

            for id, limb in enumerate(handLimbs.landmark):  # Enumerate between nodes on hands
                height, width, x = img.shape
                convertX, convertY = int(limb.x * width), int(limb.y * height)  # Frame size ratio to pixel placement
                # print("ID: " + str(id) + " | POS:" + " x=" + str(convertX) + " y=" + str(convertY))

                # Draw circles at landmarks
                cv2.circle(img, (convertX, convertY), 10, (id * int(255 / 21), id * int(255 / 21), id * int(255 / 21)), cv2.FILLED)
                sumx += convertX
                sumy += convertY
                globCalcX.append(convertX)
                globCalcY.append(convertY)
            # Draw lines between landmarks
            _mediaPipeDraw.draw_landmarks(img, handLimbs, _mediaPipeHands.HAND_CONNECTIONS)

    if didCalc and didPredict:
        sum = 0
        for px, py, cx, cy in zip(globPredX, globPredY, globCalcX, globCalcY):
            sum += (((px - cx)/1920.0) ** 2 + ((py - cy)/1080.0) ** 2) / 42.0
            # print(px, py, cx, cy)
        print("MSE: ", sum)
    
    cv2.circle(pred_img, (sumx, sumy), 20, (1, 0, 1), cv2.FILLED)

    globPredX = []
    globPredY = []
    globCalcX = []
    globCalcY = []

    # FPS Calculation
    _cTime = time.time()
    fps = 1/(_cTime - _pTime)
    _pTime = _cTime
    cv2.putText(img, "Manual Tracking Overide: " + "True" + " / FPS: " + str("{:.2f}".format(fps)) if trackBool == False else "Manual Tracking Overide: False" + " / FPS: " + str("{:.2f}".format(fps)), (10, 30), cv2.FONT_ITALIC, 1, (255, 255, 255), 1)
    # Show frame
    cv2.imshow("_videoCapture - IMG", img)
    cv2.waitKey(1)
    didPredict = False
    didCalc = False
