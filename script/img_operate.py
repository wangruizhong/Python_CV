# Prepares the library.
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt


ROOT_DIR = os.path.join(os.environ["HOME"], "python_opencv", "Python_CV")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
VIDEO_DIR = os.path.join(ROOT_DIR, "video")


def img_op_cv():
    filename = os.path.join(IMAGE_DIR, "veh.png")

    img = cv.imread(filename)
    # Name the title of window.
    cv.namedWindow("image", cv.WINDOW_NORMAL)
    # Save the image as another name.
    save_file = os.path.join(IMAGE_DIR, "veh_copy.png")
    # Show images.
    cv.imshow("image", img)

    # Set the program quit type.

    k = cv.waitKey(0)
    if k == 27:
        cv.destroyAllWindows()
    elif k == ord('s'):
        cv.imwrite(save_file, img)
        cv.destroyAllWindows()


def img_op_matplot():

    img_path = os.path.join(IMAGE_DIR, "forest.png")
    img = cv.imread(img_path, 0)

    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def use_cap():
    # Use the camera.
    cap = cv.VideoCapture(0)
    while(True):
        # Capture frame by frame.
        ret, frame = cap.read()

        # The operations on the frame come here.
        # Trans RGB image to Gray.
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Display the resulting frame.
        cv.imshow("vedio", gray)
        if cv.waitKey(1) == ord('q'):
            # 如果waitKey中的参数是0，程序会一直等待，导致画面不会更新
            # 这里设置为1，是为了让程序等待1ms
            break
    cap.release()
    cv.destroyAllWindows()


def read_vedio():
    video_path = os.path.join(VIDEO_DIR, "driving.avi")
    cap = cv.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow("frame", gray)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


def flip_video():
    cap = cv.VideoCapture(0)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    output_path = os.path.join(VIDEO_DIR, "flip_video.avi")
    out = cv.VideoWriter(output_path, fourcc, 20.0, (640, 480))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv.flip(frame, 0)
            # Write the flipped frame.
            cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()
    cv.destroyAllWindows()


def draw_line(start, end):
    img = np.zeros((512, 512, 3), np.uint8)
    cv.line(img, start, end, (255, 0, 0), 5)
    cv.imshow("line", img)
    if cv.waitKey(0) == ord('q'):
        cv.destroyAllWindows()


def draw_rectangle(shape, color):
    pt1, pt2 = shape
    print((pt1))
    img = np.zeros((512, 512, 3), np.uint8)
    cv.rectangle(img, pt1, pt2, color, 3)
    cv.imshow("rectangle", img)
    if cv.waitKey(0) == ord('q'):
        cv.destroyAllWindows()


def draw_ellipse(center, axes, angle, startAngle, endAngle, color):
    img = np.zeros((512, 512, 3), np.uint8)
    cv.ellipse(img, center, axes, angle, startAngle, endAngle, color)
    cv.imshow("ellipse", img)
    if cv.waitKey(0) == ord('q'):
        cv.destroyAllWindows()


def draw_polylines(points, color):
    pts = np.array(points, np.int32)
    pts = pts.reshape(-1, 1, 2)

    img = np.zeros((512, 512, 3), np.uint8)
    cv.polylines(img, [pts], True, color)
    cv.imshow("polylines", img)
    if cv.waitKey(0) == ord('q'):
        cv.destroyAllWindows()


def draw_text(contents, position, fontScale, color):
    font = cv.FONT_HERSHEY_SIMPLEX
    img = np.zeros((512, 512, 3), np.uint8)
    cv.putText(img, contents, position, font,
               fontScale, color, 2, cv.LINE_AA)
    cv.imshow("text", img)
    if cv.waitKey(0) == ord('q'):
        cv.destroyAllWindows()


def draw_circle():
    pass


def draw_by_click(event, x, y, flags, param):

    radius = 10

    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img, (x, y), radius, color, -1)
    cv.namedWindow("ClickEvent")
    cv.setMouseCallback("ClickEvent", draw_by_click)
    while(1):
        cv.imshow("ClickEvent", img)
        if cv.waitKey(0) == ord('q'):
            break
    cv.destroyAllWindows()


def nothing():
    pass


def slide_bar():
    cv.namedWindow('color')

    cv.createTrackbar('R', 'color', 0, 255, nothing)
    cv.createTrackbar('G', 'color', 0, 255, nothing)
    cv.createTrackbar('B', 'color', 0, 255, nothing)

    switch = '0 : OFF \n 1 : ON'
    cv.createTrackbar(switch, 'color', 0, 1, nothing)
    print("work")
    while(1):
        cv.imshow('color', img)
        k = cv.waitKey(1)
        if k == 27:
            break
        r = cv.getTrackbarPos('R', 'color')
        g = cv.getTrackbarPos('G', 'color')
        b = cv.getTrackbarPos('B', 'color')
        s = cv.getTrackbarPos(switch, 'color')

        if s == 0:
            img[:] = 0
        else:
            img[:] = [b, g, r]

    cv.destroyAllWindows()


def pixel():
    filename = os.path.join(IMAGE_DIR, "messi.png")
    img = cv.imread(filename)
    px = img[100, 100]
    print(px)
    blue = img[100, 100, 0]
    print("blue", blue)
    print("item function:", img.item(100, 100, 0))
    img.itemset((100, 100, 0), 100)
    print("item function:", img.item(100, 100, 0))
    print("img shape:", img.shape)
    print("img_size:", img.size)
    print("img_dtype:", img.dtype)
    ball = img[180:240, 330:390]
    print("ball_shape", ball.shape)
    img[173:233, 100:160] = ball
    cv.imshow('messi', img)
    cv.waitKey(0)


def merge_img():
    filename = os.path.join(IMAGE_DIR, "messi.png")
    img = cv.imread(filename)

    cv.imshow("origin", img)
    b, g, r = cv.split(img)
    img = cv.merge((g, b, r))
    cv.imshow("current", img)
    img[:, :, 2] = 0
    cv.imshow("change", img)
    cv.waitKey(0)


def add_img():
    img1_path = os.path.join(IMAGE_DIR, "messi.png")
    img2_path = os.path.join(IMAGE_DIR, "road.png")
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)

    img2 = cv.resize(img2, (150, 50))
    print("img1_shape:", img1.shape)
    print("img2_shape:", img2.shape)

    rows, cols, channels = img2.shape

    roi = img1[0:rows, 0:cols]

    img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)

    img1_bg = cv.bitwise_not(roi, roi, mask=mask_inv)

    img2_fg = cv.bitwise_and(img2, img2, mask=mask)

    dst = cv.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst
    cv.imshow("res", img1)
    cv.waitKey(0)
    cv.destroyAllWindows()


def BGR2HSV():
    cap = cv.VideoCapture(0)

    while(1):
        _, frame = cap.read()
        # Convert BGR to HSV.
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Define range of blue color in hsv.
        lowerb = np.array([110, 50, 50])
        upperb = np.array([130, 255, 255])

        # Threshold the HSV image to get only blue colors.
        mask = cv.inRange(hsv, lowerb, upperb)

        # Bitwise-and mask and original image.
        res = cv.bitwise_and(frame, frame, mask=mask)

        cv.imshow('frame', frame)
        cv.imshow('mask', mask)
        cv.imshow('res', res)
        if cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()


def transform():
    # Image transform.
    img_path = os.path.join(IMAGE_DIR, "messi.png")
    img = cv.imread(img_path)
    # res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    # OR
    # height, width = img.shape[:2]
    # res = cv.resize(img, (2*width, 2*height), interpolation=cv.INTER_CUBIC)
    # cv.imshow('original', img)
    # cv.imshow('current', res)

    # Move.
    rows, cols, ch = img.shape
    # M = np.float32([[1, 0, 100], [0, 1, 50]])
    # dst = cv.warpAffine(img, M, (cols, rows))
    # cv.imshow('Move', dst)

    # Rotation.
    # M = cv.getRotationMatrix2D((cols/2, rows/2), 90, 1)
    # dst = cv.warpAffine(img, M, (cols, rows))
    # cv.imshow('Rotation', dst)
    # if cv.waitKey(0) == ord('q'):
    #     cv.destroyAllWindows()

    # Shape change.
    # pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    # pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

    # M = cv.getAffineTransform(pts1, pts2)
    # dst = cv.warpAffine(img, M, (cols, rows))

    # Perspective Transform.
    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, M, (300, 300))

    plt.subplot(121)
    plt.imshow(img)
    plt.title('Input')
    plt.subplot(122)

    plt.imshow(dst)
    plt.title('Output')
    plt.show()


# Main.
if __name__ == "__main__":
    # img_op_cv()
    # img_op_matplot()
    # use_cap()
    # read_vedio()
    # flip_video()
    # draw_line((0, 0), (512, 512))
    shape = [(384, 0), (510, 128)]
    color = (np.random.randint(0, 255), np.random.randint(
        0, 255), np.random.randint(0, 255))
    # draw_rectangle(shape, color)
    # draw_ellipse((256,256), (100,50), 0, 0, 180, color)
    points = [[0, 0], [10, 20], [40, 80], [
        80, 80], [70, 100], [60, 70], [255, 266]]
    # draw_polylines(points, color)

    # draw_text("OpenCV", (5, 100), 4, color)
    # img = np.zeros((512, 512, 3), np.uint8)
    # events = [i for i in dir(cv) if 'EVENT' in i]
    # draw_by_click(cv.EVENT_LBUTTONDBLCLK, 0, 0, 0, 0)
    # slide_bar()
    # pixel()
    # merge_img()
    # add_img()
    # BGR2HSV()
    transform()
