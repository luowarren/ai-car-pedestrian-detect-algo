from locale import YESEXPR
from pickle import FRAME
import cv2

# Our image
img_file = "car_image.jpg"
# video_file = "videoxdxd.mp4"
video_file = "video6xd.mp4"
# video_file = "video2xd.mp4"

"""
# create opencv image (reads the image pixels and transfers into data as m-d arrays)
img = cv2.imread(img_file)

# convert to grayscale 
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

"""
# Our pre-trained car classfier
classifier_file = "car_detector.xml"
pedestrian_file = "pedestrian_detector.xml"


# video
video = cv2.VideoCapture(video_file)

# create classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

pedestrian_tracker = cv2.CascadeClassifier(pedestrian_file)


while True:
    # reading the current frame
    (read_successful, frame) = video.read()
    if not read_successful:
        break

    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_tracker.detectMultiScale(gray_scale)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img=frame, text='CAR', org=(x + 5, y + 15), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 255, 0),thickness=1)

    """
    pedestrians = pedestrian_tracker.detectMultiScale(grxay_scale)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img=frame, text='PERSON', org=(x + 5, y + 15), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 0, 0),thickness=1)
    """

    cv2.imshow('AI Real Time Car & Pedestrian Detect', frame)

    cv2.waitKey(1)



"""

# create classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# detect cars (coordinates or cars)
cars = car_tracker.detectMultiScale(black_n_white)

# draw boxes
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)



# display the image with cars spotted
cv2.imshow('Clever Programmer Car Detector', img)

# dont autoclose (Wait here in the code and listen for a key press)
cv2.waitKey()

"""

print("Code completed")