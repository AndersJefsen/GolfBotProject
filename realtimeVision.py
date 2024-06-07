import cv2
import numpy as np

def draw_dot(image, position, label, color=(0, 255, 255)):
    cv2.circle(image, position, 5, color, -1)
    cv2.putText(image, label, (position[0] + 10, position[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def find_balls(image, threshold=230):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours




def process_field(frame):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv_image, np.array([0, 120, 70]), np.array([10, 255, 255])) + \
               cv2.inRange(hsv_image, np.array([170, 120, 70]), np.array([180, 255, 255]))
    edges = cv2.Canny(red_mask, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("before contours")
    
    corners_found = False  # Flag to indicate if corners are found

    if contours:
        print("Found contours")
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.009 * cv2.arcLength(largest_contour, True)
        approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approx_corners) == 4:
            corners_found = True
            cv2.polylines(frame, [approx_corners], True, (255, 0, 0), 3)
            center_x = int(sum([corner[0][0] for corner in approx_corners]) / 4)
            center_y = int(sum([corner[0][1] for corner in approx_corners]) / 4)
            center = (center_x, center_y)
            sorted_corners = sorted(approx_corners[:, 0, :], key=lambda x: (x[1], x[0]))
            top_corners = sorted(sorted_corners[:2], key=lambda x: x[0])
            bottom_corners = sorted(sorted_corners[2:], key=lambda x: x[0], reverse=True)
            draw_dot(frame, tuple(top_corners[0]), 'TL (0,120)')
            draw_dot(frame, tuple(top_corners[1]), 'TR (180,120)')
            draw_dot(frame, tuple(bottom_corners[0]), 'BR (180,0)')
            draw_dot(frame, tuple(bottom_corners[1]), 'BL (0,0)')
            draw_dot(frame, center, 'Center')

    ball_contours = find_balls(frame)
    for i, contour in enumerate(ball_contours, 1):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if corners_found:
            # Perform calculations only if corners were found
            real_x = (x - bottom_corners[1][0]) / (top_corners[1][0] - bottom_corners[1][0]) * 180
            real_y = 120 - (y - top_corners[0][1]) / (bottom_corners[1][1] - top_corners[0][1]) * 120
            print(f"Ball {i} at: ({real_x:.2f}cm, {real_y:.2f}cm)")

    return frame




def analyze_video():
    cap = cv2.VideoCapture(1)  # Use 1 or the correct index for your camera

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from webcam.")
            break

        # Detect color and shape
        processed_frame = process_field(frame)
        print("showing frame")
        # Display the processed frame
        cv2.imshow('Processed Frame', processed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    analyze_video()



