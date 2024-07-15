import face_recognition
import cv2
import numpy as np
from datetime import datetime

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    raise Exception("Could not open video device")

sounarva_image = face_recognition.load_image_file("sounarva.jpg")
sounarva_face_encoding = face_recognition.face_encodings(sounarva_image)[0]

souradip_image = face_recognition.load_image_file("souradip.jpg")
souradip_face_encoding = face_recognition.face_encodings(souradip_image)[0]

dibyasree_image = face_recognition.load_image_file("dibyasree.jpg")
dibyasree_face_encoding = face_recognition.face_encodings(dibyasree_image)[0]

subhasree_image = face_recognition.load_image_file("subhasree.jpg")
subhasree_face_encoding = face_recognition.face_encodings(subhasree_image)[0]

sayantan_image = face_recognition.load_image_file("sayantan.jpg")
sayantan_face_encoding = face_recognition.face_encodings(sayantan_image)[0]

swarna_image = face_recognition.load_image_file("swarna.jpg")
swarna_face_encoding = face_recognition.face_encodings(swarna_image)[0]

known_face_encodings = [
    sounarva_face_encoding,
    souradip_face_encoding,
    dibyasree_face_encoding,
    subhasree_face_encoding,
    sayantan_face_encoding,
    swarna_face_encoding
]

known_face_details = {
    "Sounarva Bardhan": {"roll_number": "500421010079", "year": "3rd", "semester": "6th"},
    "Souradip Das": {"roll_number": "500421010080", "year": "3rd", "semester": "6th"},
    "Dibyasree Basu": {"roll_number": "500421020033", "year": "3rd", "semester": "6th"},
    "Subhasree Dhar": {"roll_number": "500421020075", "year": "3rd", "semester": "6th"},
    "Sayantan Maity": {"roll_number": "500421010071", "year": "3rd", "semester": "6th"},
    "Swarna Banerjee": {"roll_number": "500421020088", "year": "3rd", "semester": "6th"},
}

known_face_names = list(known_face_details.keys())

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
recognized_once = {name: False for name in known_face_names}

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        if name == "Unknown":
            box_color = (0, 0, 255)  
            label_text_color = (255, 255, 255) 
        else:
            box_color = (0, 255, 0) 
            label_text_color = (0, 0, 0)

        box_thickness = 3 
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, box_thickness)

        if name in known_face_details:
            details = known_face_details[name]
            label = f"{name}\nRoll No: {details['roll_number']}\nYear: {details['year']}\nSemester: {details['semester']}"
        else:
            label = name

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        y_offset = bottom + 20

        for i, line in enumerate(label.split('\n')):
            label_size, _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            label_width, label_height = label_size

            
            x_offset = left + (right - left - label_width) // 2

            
            cv2.putText(frame, line, (x_offset, y_offset + label_height + 2), font, font_scale, label_text_color, font_thickness)
            y_offset += label_height + 10

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    num_faces = len(face_locations)
    cv2.putText(frame, f'Faces: {num_faces}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()