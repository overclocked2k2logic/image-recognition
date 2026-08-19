import cv2
import numpy as np

def detect_aruco_webcam():
    # 1. Initialize VideoCapture for USB webcam (index 1)
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # 2. Define the Original ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    
    # 3. Create detector parameters and the detector object
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    print("Starting ArUco detection. Press 'q' to quit.")

    while True:
        # 4. Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # 5. Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 6. Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)

        # 7. Draw detected markers on the original frame
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            print(f"Detected IDs: {ids.flatten()}")

        # 8. Display the result
        cv2.imshow('ArUco Detection', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_aruco_webcam()