import cv2 as cv
import numpy as np
from model import Net
import torch

# The video feed is read in as
# a VideoCapture object
cap = cv.VideoCapture("kettlebell.mov")

# ret = a boolean return value from
# getting the frame, first_frame = the
# first frame in the entire video sequence
ret, first_frame = cap.read()
first_frame = cv.resize(first_frame, (32, 32))

# Converts frame to grayscale because we
# only need the luminance channel for
# detecting edges - less computationally
# expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

# Creates an image filled with zero
# intensities with the same dimensions
# as the frame
mask = np.zeros_like(first_frame)

# Sets image saturation to maximum
mask[..., 1] = 255

count = 0

model = Net()
model.load_state_dict(torch.load("model_weights/model.pth"))
model.eval()

prediction_map = {0: "NOTHING", 1: "DOWN", 2: "UP"}

prev_position = "NOTHING"
up_count = 0
total_count = 0

while (cap.isOpened()):

    # ret = a boolean return value from getting
    # the frame, frame = the current frame being
    # projected in the video
    ret, frame = cap.read()

    # Opens a new window and displays the input
    # frame
    cv.imshow("input", frame)

    frame = cv.resize(frame, (32, 32))

    # Converts each frame to grayscale - we previously
    # only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


    # Calculates dense optical flow by Farneback method
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
                                       None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)

    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # Sets image hue according to the optical flow
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2

    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    model_input = rgb.transpose(2, 0, 1)
    model_input = torch.from_numpy(model_input).unsqueeze(0)

    # Opens a new window and displays the output frame
    cv.imshow("dense optical flow", rgb)


    # Updates previous frame
    prev_gray = gray

    # Frames are read by intervals of 1 millisecond. The
    # programs breaks out of the while loop when the
    # user presses the 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


    # run through model
    model_input = model_input.float()
    output = model(model_input)
    pred = output.max(1)[1]
    position = prediction_map[pred.item()]

    if prev_position == "UP":
        up_count += 1
    else:
        up_count = 0

    if up_count == 8:
        total_count += 1
        print(f"SWING COUNT = {total_count}")

    prev_position = position


# The following frees up resources and
# closes all windows
cap.release()
cv.destroyAllWindows()