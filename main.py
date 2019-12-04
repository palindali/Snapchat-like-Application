#not tried
#for face detection then landmark detection

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

if __name__ == "__main__":
    cap = cv.VideoCapture(0)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    sticker = cv.imread("flowers.png", cv.IMREAD_UNCHANGED)
    print(sticker.shape)
    height, width, _ = sticker.shape
    ratio = height / width
    # bgr = sticker[:, :, :3]  # Channels 0..2
    # gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    # bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    # alpha = sticker[:, :, 3]  # Channel 3
    # result = np.dstack([bgr, alpha])  # Add the alpha channel

    alpha = 0.5
    while True:
        _, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # print(frame[0: 0 + height, 0: 0 + width, :].shape)


        faces = detector(gray)
        flower_stickers = []
        for x in range(len(faces)):
            flower_stickers.append(sticker)
        i = 0
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

            xx = landmarks.part(1).y - landmarks.part(17).y
            flower_stickers[i] = image_resize(sticker, width=xx*7)
            st_height, st_width, _ = flower_stickers[i].shape
            overlay_transparent(frame, flower_stickers[i], max(0, int(x - st_width//2)), max(0, int(y - st_height//2 -100)))

            # weighted = cv.addWeighted(frame[0:0 + height, 0:0 + width, :], alpha, sticker[0:height, 0:width, :], 1 - alpha, 0)
            # frame[0:0 + height, 0:0 + width, :] = weighted
            i += 1
        
        cv.imshow("Frame", frame)

        key = cv.waitKey(1)
        if key == 27:
            break
    pass
