import cv2
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


class ImageProcessor:
    def __init__(self):
        pass

    @staticmethod
    def load_image(image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load from {image_path}")
            return None
        return image


    @staticmethod
    def find_balls_hsv(image, min_size=300, max_size=1000000000, white_area_size=800, padding=15, min_size2=100): #Størrelsen af farven hvid der skal findes
        # Coneert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for white color in HSV
        white_lower = np.array([0, 0, 200], dtype="uint8")
        white_upper = np.array([180, 60, 255], dtype="uint8")

        # Threshhold the HSV image to get only white colors
        white_mask = cv2.inRange(hsv_image, white_lower, white_upper)

        # Use morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)


        # Load maskerne på billedet
        cv2.imshow('Processed Image Balls', white_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Find contours
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ball_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            print(f"Contour area: {area}")
            if min_size <= area < 10000:
                print(f"Inside min_size <= area <= max_size block for area: {area}")
                if area > white_area_size:
                    print("Entering area > white_area_size block")
                    # Create subplots with 1 row and 2 columns
                    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

                    # Extract the region of interest
                    x, y, w, h = cv2.boundingRect(cnt)
                    x_pad = max(x - padding, 0)
                    y_pad = max(y - padding, 0)
                    w_pad = min(w + 2 * padding, image.shape[1] - x_pad)
                    h_pad = min(h + 2 * padding, image.shape[0] - y_pad)
                    sub_image = white_mask[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]

                    # sure background area
                    sure_bg = cv2.dilate(sub_image, kernel, iterations=3)
                    axes[0, 0].imshow(sure_bg, cmap='gray')
                    axes[0, 0].set_title('Sure Background')

                    # Distance transform
                    dist = cv2.distanceTransform(sub_image, cv2.DIST_L2, 0)
                    axes[0, 1].imshow(dist, cmap='gray')
                    axes[0, 1].set_title('Distance Transform')

                    # foreground area
                    ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
                    sure_fg = sure_fg.astype(np.uint8)
                    axes[1, 0].imshow(sure_fg, cmap='gray')
                    axes[1, 0].set_title('Sure Foreground')

                    # unknown area
                    unknown = cv2.subtract(sure_bg, sure_fg)
                    axes[1, 1].imshow(unknown, cmap='gray')
                    axes[1, 1].set_title('Unknown')

                    plt.show()

                    # Marker labelling
                    # sure foreground
                    ret, markers = cv2.connectedComponents(sure_fg)

                    # Add one to all labels so that background is not 0, but 1
                    markers += 1

                    # mark the region of unknown with zero
                    # markers[unknown == 255] = 0

                    # Apply watershed
                    sub_image_color = cv2.cvtColor(sub_image, cv2.COLOR_GRAY2BGR)
                    markers = cv2.watershed(cv2.cvtColor(sub_image, cv2.COLOR_GRAY2BGR), markers)

                    markers[markers == -1] = 0  # Set the borders to 0

                    sub_image_color[markers > 1] = [0, 165, 255]  # Orange color for segmented regions

                    hsv_image = cv2.cvtColor(sub_image_color, cv2.COLOR_BGR2HSV)

                    # Define range for orange color in HSV
                    orange_lower = np.array([15, 100, 20], dtype="uint8")
                    orange_upper = np.array([25, 255, 255], dtype="uint8")

                    # Threshhold the HSV image to get only white colors
                    orange_mask = cv2.inRange(hsv_image, orange_lower, orange_upper)

                    sub_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Display the segmented image
                    cv2.imshow('Watershed Segmented Image', orange_mask)
                    cv2.waitKey(0)

                    for sub_cnt in sub_contours:
                        sub_area = cv2.contourArea(sub_cnt)
                        print("iaosidjaosijdosi")
                        print(f"sub area: {sub_area}")
                        if min_size2 <= sub_area:
                            print("I SHITTING")
                            perimeter = cv2.arcLength(sub_cnt, True)
                            if perimeter == 0:
                                print("Perimeter")
                                continue
                            circularity = 4 * np.pi * (sub_area / (perimeter * perimeter))
                            if 0.4 <= circularity <= 1.5:
                                print("Circularity")
                                sub_cnt = sub_cnt + np.array([[x_pad, y_pad]])
                                ball_contours.append(sub_cnt)

                else:
                    print("Entering else block for singular balls")
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0:
                        continue
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if 0.7 <= circularity <= 1.2:
                        ball_contours.append(cnt)
        """"
        # Logikken for at finde countours på boldene
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_size <= area <= max_size:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if 0.7 <= circularity <= 1.2:
                    ball_contours.append(cnt)
        """
        output_image = image.copy()

        cv2.drawContours(output_image, ball_contours, -1, (0,255,0),2)

        return ball_contours, output_image

    @staticmethod
    def find_orangeballs_hsv(image, min_size=300, max_size=1000000000):
        # Coneert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for orange color in HSV
        orange_lower = np.array([150, 50, 20], dtype="uint8")
        orange_upper = np.array([250, 200, 90], dtype="uint8")

        # Threshhold the HSV image to get only white colors
        orange_mask = cv2.inRange(hsv_image, orange_lower, orange_upper)

        # Use morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)

        # Load maskerne på billedet
        cv2.imshow('Processed Image', orange_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Find contours
        contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        orangeball_contours = []
        # Logikken for at finde countours på boldene
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_size <= area <= max_size:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if 0.7 <= circularity <= 1.2:
                    orangeball_contours.append(cnt)
        return orangeball_contours

    @staticmethod
    def find_robot(image, min_size=0, max_size=100000):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue_lower = np.array([105, 100, 100], dtype="uint8")
        blue_upper = np.array([131, 255, 255], dtype="uint8")

        # Threshold the HSV image to get only blue colors
        blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)

        # Use morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

        cv2.imshow('Processed Image Robot', blue_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Find contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None, image

        robot_counters = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_size or area > max_size:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.7 <= circularity <= 1.2:  # Filter round shapes based on circularity
                robot_counters.append(cnt)

        if len(robot_counters) == 0:
            print("No round contours found.")
            return None, image

        # Sort the round contours by area and select the three largest
        robot_counters = sorted(robot_counters, key=cv2.contourArea, reverse=True)[:3]

        return robot_counters, image

    @staticmethod
    def convert_to_cartesian(pixel_coords, bottom_left, bottom_right, top_left, top_right):
        x_scale = 180 / max(bottom_right[0] - bottom_left[0], top_right[0] - top_left[0])
        y_scale = 120 / max(bottom_left[1] - top_left[1], bottom_right[1] - top_right[1])
        x_cartesian = (pixel_coords[0] - bottom_left[0]) * x_scale
        y_cartesian = 120 - (pixel_coords[1] - top_left[1]) * y_scale
        x_cartesian = max(min(x_cartesian, 180), 0)
        y_cartesian = max(min(y_cartesian, 120), 0)
        return x_cartesian, y_cartesian

    @staticmethod
    def detect_all_corners(filtered_contours, image_width, image_height):
        corners = []
        for cnt in filtered_contours:
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            corners.extend(approx)

        corners.sort(key=lambda point: point[0][0] + point[0][1])
        top_left_corner = corners[0][0]
        bottom_left_corner = corners[1][0]
        top_right_corner = corners[2][0]
        bottom_right_corner = corners[3][0]
        top_left_corner = (max(0, top_left_corner[0]), max(0, top_left_corner[1]))
        bottom_left_corner = (max(0, bottom_left_corner[0]), min(image_height, bottom_left_corner[1]))
        top_right_corner = (min(image_width, top_right_corner[0]), max(0, top_right_corner[1]))
        bottom_right_corner = (min(image_width, bottom_right_corner[0]), min(image_height, bottom_right_corner[1]))
        return bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner

    @staticmethod
    def calculate_scale_factors(bottom_left, bottom_right, top_left, top_right):
        bottom_width = np.linalg.norm(np.array(bottom_left) - np.array(bottom_right))
        top_width = np.linalg.norm(np.array(top_left) - np.array(top_right))
        left_height = np.linalg.norm(np.array(bottom_left) - np.array(top_left))
        right_height = np.linalg.norm(np.array(bottom_right) - np.array(top_right))
        x_scale = 180 / max(bottom_width, top_width)
        y_scale = 120 / max(left_height, right_height)
        return x_scale, y_scale

    @staticmethod
    def find_cross_contours(filtered_contours, image):
        found_cross = False
        cross_contours = []
        for cnt in filtered_contours:
            approx = cv2.approxPolyDP(cnt, 0.015 * cv2.arcLength(cnt, True), True) # justere efter billede
            print(f"Contour length: {len(approx)}")  # Debug statement
            if len(approx) == 12:  # Our cross has 12 corners.
                bounding_rect = cv2.boundingRect(cnt)
                aspect_ratio = bounding_rect[2] / float(bounding_rect[3])
                print(f"Aspect ratio: {aspect_ratio}")  # Debug statement
                if 0.8 <= aspect_ratio <= 1.2:  # Bounds
                    if not found_cross:  # Locate it.
                        cross_contours.append(approx)
                        found_cross = True  # The cross got found!
                        for i, point in enumerate(approx):
                            x, y = point.ravel()
                            cv2.putText(image, str(i + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    else:
                        break  # Stop searching after a cross once found.
        if not found_cross:
            print("Cross not found.")  # Debug statement
        output_image = image.copy()
        return cross_contours, output_image

    @staticmethod
    def find_arena(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        red = cv2.threshold(lab[:, :, 1], 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        edges = cv2.Canny(red, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Show the mask used to find the arena contours
        cv2.imshow('Arena Mask', red)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(contours) == 0:
            print("No contours found in arena.")  # Debug statement
            return [], image

        max_contour = max(contours, key=cv2.contourArea)
        max_contour_area = cv2.contourArea(max_contour) * 0.99
        min_contour_area = cv2.contourArea(max_contour) * 0.002

        filtered_contours = [cnt for cnt in contours if max_contour_area > cv2.contourArea(cnt) > min_contour_area]
        for cnt in filtered_contours:
            font = cv2.FONT_HERSHEY_COMPLEX
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
            cv2.drawContours(image, [approx], 0, (60, 0, 0), 5)
        output_image = image.copy()
        cv2.drawContours(output_image, filtered_contours, -1, (60, 0, 0), 3)

        print(f"Filtered contours: {len(filtered_contours)}")  # Debug statement
        return filtered_contours, output_image

    @staticmethod
    def convert_cross_to_cartesian(cross_contours, image, bottom_left_corner, bottom_right_corner, top_left_corner,
                                   top_right_corner):
        cartesian_coords_list = []
        output_image = image.copy()

        for i, cnt in enumerate(cross_contours):
            for point in cnt:
                x, y = point.ravel()
                cv2.circle(output_image, (x, y), 5, (0, 0, 255), -1)
            if bottom_left_corner is not None:
                cartesian_coords = [ImageProcessor.convert_to_cartesian((point[0][0], point[0][1]), bottom_left_corner,
                                                                        bottom_right_corner, top_left_corner,
                                                                        top_right_corner) for point in cnt]
                cartesian_coords_list.append(cartesian_coords)
                print(f"Cross {i + 1} Cartesian Coordinates:")
                for coord in cartesian_coords:
                    print(f"    {coord}")
        return cartesian_coords_list, output_image
    @staticmethod
    def convert_balls_to_cartesian(image,ball_contours,bottom_left_corner,bottom_right_corner,top_right_corner,top_left_corner):
        cartesian_coords_list =[]
        output_image = image.copy()

        for i, contour in enumerate(ball_contours, 1):
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            if bottom_left_corner is not None:
                cartesian_coords = ImageProcessor.convert_to_cartesian((center_x, center_y), bottom_left_corner,
                                                                       bottom_right_corner, top_left_corner,
                                                                       top_right_corner)
                print(f"Ball {i} Cartesian Coordinates: {cartesian_coords}")

            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_image, f"{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return cartesian_coords_list, output_image

    @staticmethod
    def convert_robot_to_cartesian(image, robot_contours, bottom_left_corner, bottom_right_corner, top_left_corner,
                                   top_right_corner):
        robot_coordinates = []
        output_image = image.copy()

        if robot_contours is None or len(robot_contours) < 3:
            print("Not enough blue dots found.")
            return robot_coordinates, output_image

        for cnt in robot_contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(output_image, (cX, cY), 5, (0, 255, 0), -1)  # Mark the blue dots on the image
                if bottom_left_corner is not None:
                    cartesian_coords = ImageProcessor.convert_to_cartesian((cX, cY), bottom_left_corner,
                                                                           bottom_right_corner, top_left_corner,
                                                                           top_right_corner)
                    robot_coordinates.append(cartesian_coords)
                    print(f"Robot Cartesian Coordinates: {cartesian_coords}")

        return robot_coordinates, output_image

    @staticmethod
    def process_image(image):
        filtered_contours, output_image = ImageProcessor.find_arena(image)

        bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner = \
            ImageProcessor.detect_all_corners(filtered_contours, image.shape[1], image.shape[0])

        if bottom_left_corner is None or bottom_right_corner is None or top_left_corner is None or top_right_corner is None:
            print("Could not detect all corners.")
            return

        x_scale, y_scale = ImageProcessor.calculate_scale_factors(bottom_left_corner, bottom_right_corner,
                                                                  top_left_corner, top_right_corner)
        print("Bottom Left Corner - Pixel Coordinates:", bottom_left_corner)
        print("Bottom Left Corner - Cartesian Coordinates:", (round(
            ImageProcessor.convert_to_cartesian(bottom_left_corner, bottom_left_corner, bottom_right_corner,
                                                top_left_corner, top_right_corner)[0], 2), abs(round(
            ImageProcessor.convert_to_cartesian(bottom_left_corner, bottom_left_corner, bottom_right_corner,
                                                top_left_corner, top_right_corner)[1], 2))))

        print("Bottom Right Corner - Pixel Coordinates:", bottom_right_corner)
        print("Bottom Right Corner - Cartesian Coordinates:", (round(
            ImageProcessor.convert_to_cartesian(bottom_right_corner, bottom_left_corner, bottom_right_corner,
                                                top_left_corner, top_right_corner)[0], 2), abs(round(
            ImageProcessor.convert_to_cartesian(bottom_right_corner, bottom_left_corner, bottom_right_corner,
                                                top_left_corner, top_right_corner)[1], 2))))

        print("Top Left Corner - Pixel Coordinates:", top_left_corner)
        print("Top Left Corner - Cartesian Coordinates:", (round(
            ImageProcessor.convert_to_cartesian(top_left_corner, bottom_left_corner, bottom_right_corner,
                                                top_left_corner, top_right_corner)[0], 2), abs(round(
            ImageProcessor.convert_to_cartesian(top_left_corner, bottom_left_corner, bottom_right_corner,
                                                top_left_corner, top_right_corner)[1], 2))))

        print("Top Right Corner - Pixel Coordinates:", top_right_corner)
        print("Top Right Corner - Cartesian Coordinates:", (round(
            ImageProcessor.convert_to_cartesian(top_right_corner, bottom_left_corner, bottom_right_corner,
                                                top_left_corner, top_right_corner)[0], 2), abs(round(
            ImageProcessor.convert_to_cartesian(top_right_corner, bottom_left_corner, bottom_right_corner,
                                                top_left_corner, top_right_corner)[1], 2))))

        cross_contours, image_with_cross = ImageProcessor.find_cross_contours(filtered_contours, output_image)
        cartesian_coords_list, image_with_cross = ImageProcessor.convert_cross_to_cartesian(cross_contours,
                                                                                            output_image,
                                                                                            bottom_left_corner,
                                                                                            bottom_right_corner,
                                                                                            top_left_corner,
                                                                                            top_right_corner)

        # Mark the corners on the output_image
        if bottom_left_corner is not None:
            cv2.circle(output_image, bottom_left_corner, 10, (0, 0, 255), -1)
        if bottom_right_corner is not None:
            cv2.circle(output_image, bottom_right_corner, 10, (0, 255, 0), -1)
        if top_left_corner is not None:
            cv2.circle(output_image, top_left_corner, 10, (255, 135, 0), -1)
        if top_right_corner is not None:
            cv2.circle(output_image, top_right_corner, 10, (255, 0, 135), -1)

        ball_contours, image_with_balls = ImageProcessor.find_balls_hsv(output_image, min_size=300, max_size=1000)
        cartesian_coords, image_with_balls = ImageProcessor.convert_balls_to_cartesian(output_image, ball_contours,
                                                                                       bottom_left_corner,
                                                                                       bottom_right_corner,
                                                                                       top_right_corner,
                                                                                       top_left_corner)

        robot_contours, image_with_robot = ImageProcessor.find_robot(output_image, min_size=0, max_size=100000)
        if robot_contours is not None:
            robot_coordinates, image_with_robot = ImageProcessor.convert_robot_to_cartesian(output_image,
                                                                                            robot_contours,
                                                                                            bottom_left_corner,
                                                                                            bottom_right_corner,
                                                                                            top_left_corner,
                                                                                            top_right_corner)

        cv2.imshow('Final Image with cross', image_with_cross)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow('Final Image with Balls and Arena', image_with_balls)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow('Final Image with Robot', image_with_robot)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "images/Bane 2 med Gule/WIN_20240207_09_35_30_Pro.jpg"  # Path to your image
    image = ImageProcessor.load_image(image_path)
    if image is not None:
        ImageProcessor.process_image(image)
