# +++++++++++++++++++++++++++++++++++++++++++++++++++++
# Libraries
# +++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import os
import cv2
import pyautogui
import datetime
from pynput.keyboard import Key, Controller

# +++++++++++++++++++++++++++++++++++++++++++++++++++++
# Constants
# +++++++++++++++++++++++++++++++++++++++++++++++++++++
# Obstacles BBOX (400, 170) (483, 228)
OBSTACLE_BOUNDING_ORIGIN = (560, 170)
OBSTACLE_BOUNDING_END = (643, 228)

# Highest Jump BBOX (278, 14) (361, 72)
HIGHEST_JUMP_BOUNDING_ORIGIN = (438, 14)
HIGHEST_JUMP_BOUNDING_END = (521, 72)

# Keyboard Controller
KEYBOARD = Controller()

# BBOX Colors
BOUND_AREA_COLORS = {
    1: (0, 0, 255),
    0: (0, 255, 0)
}

# Up and Down Status
UP_DOWN_STATUS = {
    'up': 1,
    'down': 0
}

# Status Image BBOX
STATUS_IMAGE_Y = (540, 850)
STATUS_IMAGE_X = (0, 2080)
STATUS_IMAGE_RES = (2080, 400)

jumps = 0
prev_up_down_status = UP_DOWN_STATUS['down']

# +++++++++++++++++++++++++++++++++++++++++++++++++++++
# Functions
# +++++++++++++++++++++++++++++++++++++++++++++++++++++
# Check if there are contours
def check_contours_of_expected_area(image, area_thresh=0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 100, 250)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        area = cv2.contourArea(c)
        if area > area_thresh:
            return 1
    return 0

# Count the number of jumps
def count_jumps(cont_avb):
    global prev_up_down_status, jumps

    if cont_avb is UP_DOWN_STATUS['up']:
        if cont_avb is not prev_up_down_status:
            jumps += 1
            prev_up_down_status = cont_avb
    else:
        prev_up_down_status = 0

# +++++++++++++++++++++++++++++++++++++++++++++++++++++
# Main Loop
# +++++++++++++++++++++++++++++++++++++++++++++++++++++
# application entry

def App():
    global jumps

    cv2.waitKey(5000)

    jumps = 0

    # Start the game by pressing space
    KEYBOARD.press(Key.space)
    cv2.waitKey(1)
    KEYBOARD.release(Key.space)

    while True:
        # Capture the screen
        screen_image = np.array(pyautogui.screenshot())
        screen_image = cv2.cvtColor(screen_image, cv2.COLOR_BGR2RGB)
        screen_image = screen_image[STATUS_IMAGE_Y[0]: STATUS_IMAGE_Y[1], STATUS_IMAGE_X[0]: STATUS_IMAGE_X[1]]

        # Crop the screen image to get the front obstacle and the highest jump blocks
        front_obs_block = screen_image[
            OBSTACLE_BOUNDING_ORIGIN[1]: OBSTACLE_BOUNDING_END[1] + 1,
            OBSTACLE_BOUNDING_ORIGIN[0]: OBSTACLE_BOUNDING_END[0] + 1
        ]
        highest_jump_block = screen_image[
            HIGHEST_JUMP_BOUNDING_ORIGIN[1]: HIGHEST_JUMP_BOUNDING_END[1] + 1,
            HIGHEST_JUMP_BOUNDING_ORIGIN[0]: HIGHEST_JUMP_BOUNDING_END[0] + 1
        ]

        avb = check_contours_of_expected_area(front_obs_block)
        obstacle_area_color, obstacle_status = BOUND_AREA_COLORS[0], 0

        # If there is an obstacle, jump
        if avb:
            obstacle_area_color, obstacle_status = BOUND_AREA_COLORS[1], 1

            KEYBOARD.press(Key.space)
            cv2.waitKey(1)
            KEYBOARD.release(Key.space)

        avb = check_contours_of_expected_area(highest_jump_block)
        highest_jump_bound_area_color, jump_status = BOUND_AREA_COLORS[0], 0

        # If there is a block that requires the highest jump, jump
        if avb:
            highest_jump_bound_area_color, jump_status = BOUND_AREA_COLORS[1], 1
        count_jumps(avb)

        # Draw the bounding boxes and status on the screen image
        cv2.rectangle(screen_image,
                      OBSTACLE_BOUNDING_ORIGIN,
                      OBSTACLE_BOUNDING_END,
                      obstacle_area_color,
                      2)
        cv2.rectangle(screen_image,
                      HIGHEST_JUMP_BOUNDING_ORIGIN,
                      HIGHEST_JUMP_BOUNDING_END,
                      highest_jump_bound_area_color,
                      2)
        # cv2.rectangle(screen_image,
        #               (OBSTACLE_BOUNDING_ORIGIN[0], OBSTACLE_BOUNDING_ORIGIN[1] - 22),
        #               (OBSTACLE_BOUNDING_ORIGIN[0] + 22, OBSTACLE_BOUNDING_ORIGIN[1]),
        #               obstacle_area_color,
        #               -1)


        # Put the status and jump count on the screen image
        screen_image = cv2.putText(screen_image,
                                   'JUMPS: {}'.format(jumps),
                                   (570, 43),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   1.0,
                                   (0, 0, 255),
                                   2)
        # Place obstacle status centered inside its box
        obs_text_x = (OBSTACLE_BOUNDING_ORIGIN[0] + OBSTACLE_BOUNDING_END[0]) // 2 - 6
        obs_text_y = (OBSTACLE_BOUNDING_ORIGIN[1] + OBSTACLE_BOUNDING_END[1]) // 2 + 6
        screen_image = cv2.putText(screen_image,
                                   '{}'.format(obstacle_status),
                                   (obs_text_x, obs_text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.7,
                                   (255, 255, 255),
                                   2)
        # Place jump status centered inside its box
        jump_text_x = (HIGHEST_JUMP_BOUNDING_ORIGIN[0] + HIGHEST_JUMP_BOUNDING_END[0]) // 2 - 6
        jump_text_y = (HIGHEST_JUMP_BOUNDING_ORIGIN[1] + HIGHEST_JUMP_BOUNDING_END[1]) // 2 + 6
        screen_image = cv2.putText(screen_image,
                                   '{}'.format(jump_status),
                                   (jump_text_x, jump_text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.7,
                                   (255, 255, 255),
                                   2)

        cv2.imshow("Autoplay Dinosaur Game", screen_image)

        k = cv2.waitKey(1)
        if k == 27:
            return

App()
cv2.destroyAllWindows()