import cv2

# Load face and eye classifiers
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)

# Check that classifiers loaded correctly
if face_cascade.empty():
    print("Error loading face cascade")
    exit()

if eye_cascade.empty():
    print("Error loading eye cascade")
    exit()

# Open webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

while True:
    success, frame = cap.read()

    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        # Blue rectangle around face
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # Only search for eyes in the upper 60% of the face
        face_gray = gray[y:y + int(h * 0.6), x:x + w]
        face_color = frame[y:y + int(h * 0.6), x:x + w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20)
        )

        # Draw red rectangles around detected eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                face_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (0, 0, 255),  # Red
                2
            )

    cv2.imshow("Face and Eye Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()