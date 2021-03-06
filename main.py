import cv2 as cv
import numpy as np
import math
import dlib
import imutils
import sys

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1],
                         1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * \
        background[y:y+h, x:x+w] + mask * overlay_image

    return background

def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def placeFlower(landmarks, frame):
    sticker = cv.imread("flowers.png", cv.IMREAD_UNCHANGED)
    
    y_x = (landmarks.part(17 - 1).y - landmarks.part(1 - 1).y)/(landmarks.part(17 - 1).x - landmarks.part(1 - 1).x)
    angle = (np.arctan(y_x)*180)/np.pi
    sticker = imutils.rotate_bound(sticker, angle)

    rep_width = abs(landmarks.part(1 - 1).x - landmarks.part(17 - 1).x)
    sticker = image_resize(sticker, width=int(1.4*rep_width))


    st_height, st_width, _ = sticker.shape

    x = (landmarks.part(1 - 1).x + landmarks.part(17 - 1).x)//2
    y = landmarks.part(25 - 1).y + (landmarks.part(28 - 1).y - landmarks.part(34 - 1).y)
   
    x, y = rotate((landmarks.part(31 - 1).x,
                   landmarks.part(31 - 1).y), (x, y), np.arctan(y_x))
    x, y = int(x), int(y)
   
    st_x = x - st_width//2
    st_y = y - st_height//2

    if st_x < 0:
        sticker = sticker[:, 0 - st_x:st_width]
        st_x = 0
    if st_y < 0:
        sticker = sticker[0 - st_y:st_height, :]
        st_y = 0

    
    overlay_transparent(frame, sticker, st_x, st_y)

def placeHearts(landmarks, frame):
    sticker = cv.imread("hearts.png", cv.IMREAD_UNCHANGED)

    y_x = (landmarks.part(17 - 1).y - landmarks.part(1 - 1).y) / \
        (landmarks.part(17 - 1).x - landmarks.part(1 - 1).x)
    angle = (np.arctan(y_x)*180)/np.pi
    sticker = imutils.rotate_bound(sticker, angle)

    rep_width = abs(landmarks.part(1 - 1).x - landmarks.part(17 - 1).x)
    sticker = image_resize(sticker, width=int(1.4*rep_width))
    st_height, st_width, _ = sticker.shape

    x = abs(landmarks.part(1 - 1).x + landmarks.part(17 - 1).x)//2
    y = landmarks.part(25 - 1).y + \
        (landmarks.part(28 - 1).y - landmarks.part(34 - 1).y)
    
    x, y = rotate((landmarks.part(31 - 1).x,
                   landmarks.part(31 - 1).y), (x, y), np.arctan(y_x))
    x, y = int(x), int(y)

    st_x = x - st_width//2
    st_y = y - st_height//2
    if st_x < 0:
        sticker = sticker[:, 0 - st_x:st_width]
        st_x = 0
    if st_y < 0:
        sticker = sticker[0 - st_y:st_height, :]
        st_y = 0

    overlay_transparent(frame, sticker, st_x, st_y)

def placeLips(landmarks, frame):
    sticker = cv.imread("lips.png", cv.IMREAD_UNCHANGED)

    y_x = (landmarks.part(17 - 1).y - landmarks.part(1 - 1).y) / \
        (landmarks.part(17 - 1).x - landmarks.part(1 - 1).x)
    angle = (np.arctan(y_x)*180)/np.pi
    sticker = imutils.rotate_bound(sticker, angle)

    rep_width = abs(landmarks.part(55 - 1).x - landmarks.part(49 - 1).x)

    sticker = image_resize(sticker, width=rep_width)
    st_height, st_width, _ = sticker.shape
    x = abs( landmarks.part(63-1).x )
    y = landmarks.part(67 - 1).y

    x, y = rotate((landmarks.part(31 - 1).x,
                   landmarks.part(31 - 1).y), (x, y), np.arctan(y_x))
    x, y = int(x), int(y)

    st_x = x - st_width//2
    st_y = y - st_height//2
    if st_x < 0:
        sticker = sticker[:, 0 - st_x:st_width]
        st_x = 0
    if st_y < 0:
        sticker = sticker[0 - st_y:st_height, :]
        st_y = 0

    overlay_transparent(frame, sticker, max(0, st_x), max(0, st_y))

def placeGlasses(landmarks, frame):
    sticker = cv.imread("glasses.png", cv.IMREAD_UNCHANGED)

    y_x = (landmarks.part(17 - 1).y - landmarks.part(1 - 1).y) / \
        (landmarks.part(17 - 1).x - landmarks.part(1 - 1).x)
    angle = (np.arctan(y_x)*180)/np.pi
    sticker = imutils.rotate_bound(sticker, angle)

    # rep_width = abs(landmarks.part(27 - 1).x - landmarks.part(18 - 1).x)
    rep_width = abs(landmarks.part(17 - 1).x - landmarks.part(1 - 1).x)

    sticker = image_resize(sticker, width=rep_width)
    st_height, st_width, _ = sticker.shape
    x = landmarks.part(28 - 1).x
    y = landmarks.part(28 - 1).y

    x, y = rotate((landmarks.part(31 - 1).x,
                   landmarks.part(31 - 1).y), (x, y), np.arctan(y_x))
    x, y = int(x), int(y)

    st_x = x - st_width//2
    st_y = y - st_height//2
    if st_x < 0:
        sticker = sticker[:, 0 - st_x:st_width]
        st_x = 0
    if st_y < 0:
        sticker = sticker[0 - st_y:st_height, :]
        st_y = 0
    overlay_transparent(frame, sticker, max(0, st_x), max(0, st_y))

def placePigNose(landmarks, frame):
    sticker = cv.imread("pig.png", cv.IMREAD_UNCHANGED)

    y_x = (landmarks.part(17 - 1).y - landmarks.part(1 - 1).y) / \
        (landmarks.part(17 - 1).x - landmarks.part(1 - 1).x)
    angle = (np.arctan(y_x)*180)/np.pi
    sticker = imutils.rotate_bound(sticker, angle)

    rep_width = abs(landmarks.part(36 - 1).x - landmarks.part(32 - 1).x)

    sticker = image_resize(sticker, width=(int)(rep_width  * 1.6))
    st_height, st_width, _ = sticker.shape
    x = landmarks.part(31 - 1).x
    y = landmarks.part(31 - 1).y 

    x, y = rotate((landmarks.part(31 - 1).x,
                   landmarks.part(31 - 1).y), (x, y), np.arctan(y_x))
    x, y = int(x), int(y)

    st_x = x - st_width//2
    st_y = y - st_height//2
    if st_x < 0:
        sticker = sticker[:, 0 - st_x:st_width]
        st_x = 0
    if st_y < 0:
        sticker = sticker[0 - st_y:st_height, :]
        st_y = 0

    overlay_transparent(frame, sticker, max(0, st_x), max(0, st_y))

def placeDogNose(landmarks, frame):
    sticker = cv.imread("dogNose.png", cv.IMREAD_UNCHANGED)

    y_x = (landmarks.part(17 - 1).y - landmarks.part(1 - 1).y) / \
        (landmarks.part(17 - 1).x - landmarks.part(1 - 1).x)
    angle = (np.arctan(y_x)*180)/np.pi
    sticker = imutils.rotate_bound(sticker, angle)

    rep_width = abs(landmarks.part(36 - 1).x - landmarks.part(32 - 1).x)

    sticker = image_resize(sticker, width= int (rep_width * 1.6))
    st_height, st_width, _ = sticker.shape

    x = landmarks.part(31 - 1).x 
    y = landmarks.part(31 - 1).y 

    x, y = rotate((landmarks.part(31 - 1).x,
                   landmarks.part(31 - 1).y), (x, y), np.arctan(y_x))
    x, y = int(x), int(y)

    st_x = x - st_width//2
    st_y = y - st_height//2
    if st_x < 0:
        sticker = sticker[:, 0 - st_x:st_width]
        st_x = 0
    if st_y < 0:
        sticker = sticker[0 - st_y:st_height, :]
        st_y = 0

    overlay_transparent(frame, sticker, max(0, st_x), max(0, st_y))

def placeDogEars(landmarks, frame):
    sticker_right = cv.imread("dogEars_right.png", cv.IMREAD_UNCHANGED)
    sticker_left = cv.imread("dogEars_left.png", cv.IMREAD_UNCHANGED)

    y_x = (landmarks.part(17 - 1).y - landmarks.part(1 - 1).y) / \
        (landmarks.part(17 - 1).x - landmarks.part(1 - 1).x)
    angle = (np.arctan(y_x)*180)/np.pi
    
    sticker_right = imutils.rotate_bound(sticker_right, angle)
    sticker_left = imutils.rotate_bound(sticker_left, angle)

    rep_width = abs(landmarks.part(22 - 1).x - landmarks.part(18 - 1).x)
    sticker_right = image_resize(sticker_right, width= int (rep_width * 1.5))

    rep_width = abs(landmarks.part(23 - 1).x - landmarks.part(27 - 1).x)
    sticker_left = image_resize(sticker_left, width= int (rep_width * 1.5))

    st_height, st_width, _ = sticker_right.shape
    x = landmarks.part(18 - 1).x 
    y = landmarks.part(20 - 1).y - abs(landmarks.part(20 - 1).y - landmarks.part(2 - 1).y)
    
    x, y = rotate((landmarks.part(31 - 1).x,
                   landmarks.part(31 - 1).y), (x, y), np.arctan(y_x))
    x, y = int(x), int(y)
    
    st_x = x - st_width//2
    st_y = y - st_height//2
    if st_y < 0:
        sticker_right = sticker_right[0 - st_y:st_height, :]
        st_y = 0
    if st_x < 0:
        sticker_right = sticker_right[:, 0 - st_x:st_width]
        st_x = 0
    overlay_transparent(frame, sticker_right, max(0, st_x), max(0, st_y))

    st_height, st_width, _ = sticker_left.shape
    x = landmarks.part(27 - 1).x 
    y = landmarks.part(25 - 1).y - abs(landmarks.part(25 - 1).y - landmarks.part(16 - 1).y)
    
    x, y = rotate((landmarks.part(31 - 1).x,
                   landmarks.part(31 - 1).y), (x, y), np.arctan(y_x))
    x, y = int(x), int(y)

    st_x = x - st_width//2
    st_y = y - st_height//2
    if st_y < 0:
        sticker_left = sticker_left[0 - st_y:st_height, :]
        st_y = 0
    overlay_transparent(frame, sticker_left, max(0, st_x), max(0, st_y))
    

def placeDog(landmarks, frame):
    placeDogNose(landmarks, frame)
    placeDogEars(landmarks, frame)

def placeBeard(landmarks, face):
    sticker = cv.imread("beard.png", cv.IMREAD_UNCHANGED)

    y_x = (landmarks.part(17 - 1).y - landmarks.part(1 - 1).y) / \
        (landmarks.part(17 - 1).x - landmarks.part(1 - 1).x)
    angle = (np.arctan(y_x)*180)/np.pi
    sticker = imutils.rotate_bound(sticker, angle)

    rep_width = abs(landmarks.part(14 - 1).x - landmarks.part(4 - 1).x)
    sticker = image_resize(sticker, width=rep_width)
    st_height, st_width, _ = sticker.shape

    x = landmarks.part(9 - 1).x 
    y = landmarks.part(9 - 1).y 

    x, y = rotate((landmarks.part(31 - 1).x,
                   landmarks.part(31 - 1).y), (x, y), np.arctan(y_x))
    x, y = int(x), int(y)

    st_x = x - st_width//2
    st_y = y - st_height//2
    if st_x < 0:
        sticker = sticker[:, 0 - st_x:st_width]
        st_x = 0
    if st_y < 0:
        sticker = sticker[0 - st_y:st_height, :]
        st_y = 0

    overlay_transparent(frame, sticker, max(0, st_x), max(0, st_y))

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("Wrong parameters")
        exit()

    cap = cv.VideoCapture(0)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    while True:
        _, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = detector(gray)
        flower_stickers = []
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            # cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            landmarks = predictor(gray, face)

            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv.circle(frame, (x, y), 4, (3*n, 3*n, 0), -1)

            x = landmarks.part(31).x
            y = landmarks.part(31).y

            if (sys.argv[1] == "flower"):
                placeFlower(landmarks, frame)
            elif (sys.argv[1] == "lips"):
                placeLips(landmarks, frame)
            elif (sys.argv[1] == "hearts"):
                placeHearts(landmarks, frame)
            elif (sys.argv[1] == "pig"):
                placePigNose(landmarks, frame)
            elif (sys.argv[1] == "dog"):
                placeDog(landmarks, frame)
            elif (sys.argv[1] == "beard"):
                placeBeard(landmarks, face)
            elif (sys.argv[1] == "glasses"):
                placeGlasses(landmarks, frame)

        cv.imshow("Frame", frame)

        key = cv.waitKey(1)
        if key == 27:
            break
    pass
