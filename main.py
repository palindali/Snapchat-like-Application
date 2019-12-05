import cv2 as cv
import numpy as np
import dlib


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

def placeFlower(landmarks, frame):
    sticker = cv.imread("flowers.png", cv.IMREAD_UNCHANGED)
    height, width, _ = sticker.shape
    ratio = height / width

    xx = abs(landmarks.part(1).y - landmarks.part(17).y)
    sticker = image_resize(sticker, width=xx*7)
    st_height, st_width, _ = sticker.shape
    overlay_transparent(frame, sticker, max(0, int(x - st_width//2)), max(0, int(y - st_height//2 -150)))
    
def placeHearts(landmarks, frame):
    sticker = cv.imread("hearts.png", cv.IMREAD_UNCHANGED)
    height, width, _ = sticker.shape
    ratio = height / width

    xx = abs(landmarks.part(1).y - landmarks.part(17).y)
    sticker = image_resize(sticker, width=xx*7)
    st_height, st_width, _ = sticker.shape
    overlay_transparent(frame, sticker, max(0, int(x - st_width//2 + 10)), max(0, int(y - st_height//2 - 200)))

def placeLips(landmarks, frame):
    sticker = cv.imread("lips.png", cv.IMREAD_UNCHANGED)
    height, width, _ = sticker.shape
    ratio = height / width

    xx = abs(landmarks.part(55).y - landmarks.part(49).y)

    sticker = image_resize(sticker, width=xx*5)
    st_height, st_width, _ = sticker.shape
    overlay_transparent(frame, sticker, max(0, int(x - st_width//2 + 20)), max(0, int(y - st_height//2 + 30)))

def placeGlasses(landmarks, frame):
    sticker = cv.imread("glasses.png", cv.IMREAD_UNCHANGED)
    height, width, _ = sticker.shape
    ratio = height / width

    xx = abs(landmarks.part(27).y - landmarks.part(18).y)

    sticker = image_resize(sticker, width=xx* 8)
    st_height, st_width, _ = sticker.shape
    overlay_transparent(frame, sticker, max(0, int(x - st_width//2 + 20)), max(0, int(y - st_height//2 - 90)))

def placePigNose(landmarks, frame):
    sticker = cv.imread("pig.png", cv.IMREAD_UNCHANGED)
    height, width, _ = sticker.shape
    ratio = height / width

    xx = abs(landmarks.part(36).y - landmarks.part(32).y)

    sticker = image_resize(sticker, width=xx)
    st_height, st_width, _ = sticker.shape
    overlay_transparent(frame, sticker, max(0, int(x - st_width//2 + 15)), max(0, int(y - st_height//2 - 10)))

def placeDogNose(landmarks, frame):
    sticker = cv.imread("dogNose.png", cv.IMREAD_UNCHANGED)
    height, width, _ = sticker.shape
    ratio = height / width

    xx = abs(landmarks.part(36).y - landmarks.part(32).y)

    sticker = image_resize(sticker, width=xx)
    st_height, st_width, _ = sticker.shape
    overlay_transparent(frame, sticker, max(0, int(x - st_width//2 + 15)), max(0, int(y - st_height//2 - 10)))

def placeDogEars(landmarks, frame):
    sticker = cv.imread("dogEars.png", cv.IMREAD_UNCHANGED)
    height, width, _ = sticker.shape
    ratio = height / width

    xx = abs(landmarks.part(1).y - landmarks.part(17).y)
    sticker = image_resize(sticker, width=xx*6)
    st_height, st_width, _ = sticker.shape
    overlay_transparent(frame, sticker, max(0, int(x - st_width//2 + 20)), max(0, int(y - st_height//2 -200)))

def placeDog(landmarks, frame):
    placeDogNose(landmarks, frame)
    placeDogEars(landmarks, frame)

def placeBeard(landmarks, face):
    sticker = cv.imread("beard.png", cv.IMREAD_UNCHANGED)
    height, width, _ = sticker.shape
    ratio = height / width

    xx = abs(landmarks.part(15).y - landmarks.part(3).y)
    sticker = image_resize(sticker, width=xx*4)
    st_height, st_width, _ = sticker.shape
    overlay_transparent(frame, sticker, max(0, int(x - st_width//2 + 20)), max(0, int(y - st_height//2 + 100)))

if __name__ == "__main__":
    cap = cv.VideoCapture(0)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    

    alpha = 0.5
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
                # cv.circle(frame, (x, y), 4, (3*n, 3*n, 3*n), -1)
            
            x = landmarks.part(31).x
            y = landmarks.part(31).y

            # placeFlower(landmarks, frame)
            # placeLips(landmarks, frame)
            # placeGlasses(landmarks, frame)
            # placeHearts(landmarks, frame)
            # placePigNose(landmarks, frame)
            # placeDog(landmarks, frame)
            placeBeard(landmarks, face)

        cv.imshow("Frame", frame)

        key = cv.waitKey(1)
        if key == 27:
            break
    pass

