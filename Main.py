from keras.models import load_model
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

#-------------------------Settings------------------------

# Check and change camera port with "cameraPortTest.py"!
camera = cv2.VideoCapture(1)

# Show live webcamera image window on screen
showimage = False

# Motor pins (GPIO)
GPIO_MOTOR_R = 22
GPIO_MOTOR_L = 27
GPIO_MOTOR_2 = 17

#---------------------------------------------------------

# Initialize GPIO
GPIO.setmode(GPIO.BCM)

# Motor PWM frequency and duty cycle
PWM_FREQ = 100
PWM_DUTY_CYCLE_40 = 40
PWM_DUTY_CYCLE_70 = 70
PWM_DUTY_CYCLE_100 = 100

# Setup motors
GPIO.setup(GPIO_MOTOR_R, GPIO.OUT)
GPIO.setup(GPIO_MOTOR_L, GPIO.OUT)
GPIO.setup(GPIO_MOTOR_2, GPIO.OUT)

motor_r = GPIO.PWM(GPIO_MOTOR_R, PWM_FREQ)
motor_l = GPIO.PWM(GPIO_MOTOR_L, PWM_FREQ)
motor_2 = GPIO.PWM(GPIO_MOTOR_2, PWM_FREQ)

motor_r.start(0)
motor_l.start(0)
motor_2.start(0)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the first model
model1 = load_model("CorrModel\\Corr_model.h5", compile=False)

# Load the second model
model2 = load_model("DistModel\\Dist_model.h5", compile=False)

# Load the labels for both models
class_names_model1 = open("CorrModel\\labels.txt", "r").readlines()
class_names_model2 = open("DistModel\\labels.txt", "r").readlines()

while True:
    # Grab the web camera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    if showimage == True :
        cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models' input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the first model
    prediction_model1 = model1.predict(image)
    index_model1 = np.argmax(prediction_model1)
    class_name_model1 = class_names_model1[index_model1]
    confidence_score_model1 = prediction_model1[0][index_model1]

    # Predicts the second model
    prediction_model2 = model2.predict(image)
    index_model2 = np.argmax(prediction_model2)
    class_name_model2 = class_names_model2[index_model2]
    confidence_score_model2 = prediction_model2[0][index_model2]

    # Print predictions and confidence scores for both models
    print("Model 1 - Class:", class_name_model1[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score_model1 * 100))[:-2], "%")

    print("Model 2 - Class:", class_name_model2[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score_model2 * 100))[:-2], "%")

    ## Use the predictions to determine which loop to activate
    #if index_model1 == 0:
    #    # First loop for model1 class 1
    #    for i in range(10):
    #        print("Executing first loop for model1 class 1")
    #        # Add your code for the first loop here

    #elif index_model1 == 1:
    #    # Second loop for model1 class 2
    #    for i in range(10):
    #        print("Executing second loop for model1 class 2")
    #        # Add your code for the second loop here

    #if index_model2 == 0:
    #    # Third loop for model2 class 1
    #    for i in range(10):
    #        print("Executing third loop for model2 class 1")
    #        # Add your code for the third loop here

    #elif index_model2 == 1:
    #    # Fourth loop for model2 class 2
    #    for i in range(10):
    #        print("Executing fourth loop for model2 class 2")
    #        # Add your code for the fourth loop here

    # Use the predictions to control motors
    if class_name_model1 == "R":
        motor_r.ChangeDutyCycle(100)
        motor_l.ChangeDutyCycle(0)
    elif class_name_model1 == "L":
        motor_r.ChangeDutyCycle(0)
        motor_l.ChangeDutyCycle(100)

    if class_name_model2 == "2":
        motor_2.ChangeDutyCycle(PWM_DUTY_CYCLE_40)
    elif class_name_model2 == "3":
        motor_2.ChangeDutyCycle(PWM_DUTY_CYCLE_70)
    elif class_name_model2 == "4":
        motor_2.ChangeDutyCycle(PWM_DUTY_CYCLE_100)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
motor_r.stop()
motor_l.stop()
motor_2.stop()
GPIO.cleanup()
camera.release()
cv2.destroyAllWindows()
