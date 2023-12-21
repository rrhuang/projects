import cv2
import face_recognition
#import numpy as np
import os

known_faces = []
known_names = []

# Load images and learn how to recognize them.
for filename in os.listdir('known_people'):
    image = face_recognition.load_image_file(os.path.join('known_people',filename))
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding)
    known_names.append(os.path.splitext(filename)[0])

# Initialize a few lists for storage and recurring proess variable
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Capture video from the webcam (must have webcam, duh)
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Scale to 1/4th
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert BGR to RGB
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        print(face_locations)
        if face_locations:
            try:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                print("Success!")
            except TypeError:
                print("Error")
                face_encodings = []
        else:
            face_encodings = []
            print("No face!")
        print(face_encodings)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            face_names.append(name)
    print("Done!")
    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
