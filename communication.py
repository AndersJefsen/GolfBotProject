import cv2
import numpy as np

"""
Gamle kodet, ved ikke om det skal slettes
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load from {image_path}")
        return None
    return image

def compute_contrast_level(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #hist1 = cv2.equalizeHist(gray)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    contrast_level = np.sum(hist[64:192]) / np.sum(hist)
    return contrast_level

def process_image(image):
    contrast_level = compute_contrast_level(image)
    print("Contrast Level:", contrast_level)

    # Define base lower and upper bounds for red color detection
    lower_red_base = np.array([100, 100, 150])
    upper_red_base = np.array([130, 130, 255])

    # Adjust thresholds based on contrast level
    contrast_threshold = 0.45  # Example threshold for adjusting thresholds
    if contrast_level > contrast_threshold:
        lower_red = np.array([0, 0, 150])  # Lower threshold for lower contrast
        upper_red = np.array([100, 100, 255])
    else:
        lower_red = lower_red_base
        upper_red = upper_red_base

    # Threshold the image to get only red areas
    red_mask = cv2.inRange(image, lower_red, upper_red)

    # Find contours in the red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Ignore small contours
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Draw green contours around detected red areas

    # Show the processed image
    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "images/banemedfarve2/banemedfarve4.jpg"  # Path to your image
    image = load_image(image_path)
    if image is not None:
        process_image(image)
        
"""