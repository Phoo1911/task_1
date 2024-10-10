from PIL import Image
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.feature import canny
from skimage.measure import label, regionprops

def detect_ball(image_path):
    # Load the image using Pillow
    img = Image.open(image_path)
    img = img.convert('RGB')  # Convert to RGB

    # Convert to NumPy array and then to grayscale using skimage
    img_array = np.array(img)
    gray_img = rgb2gray(img_array)

    # Apply Gaussian blur using skimage to reduce noise
    blurred_img = gaussian(gray_img, sigma=2)

    # Use Canny edge detection from skimage
    edges = canny(blurred_img)

    # Label the connected regions in the edge-detected image
    labeled_img = label(edges)

    # Find regions (contours) in the labeled image
    regions = regionprops(labeled_img)

    # Assume the largest region is the ball
    largest_region = max(regions, key=lambda r: r.area)

    # Get the center and bounding box of the ball
    ball_center_y, ball_center_x = largest_region.centroid
    min_row, min_col, max_row, max_col = largest_region.bbox
    ball_size = max(max_row - min_row, max_col - min_col)  # Estimate the size of the ball

    # Return the ball center coordinates and size
    return (int(ball_center_x), int(ball_center_y)), ball_size

# Load and process the two images
ball_center_1, ball_size_1 = detect_ball('FirstPic.jpg')
ball_center_2, ball_size_2 = detect_ball('SecondPic.jpg')

# Print results
print(f"Ball 1 - Center: {ball_center_1}, Size: {ball_size_1}")
print(f"Ball 2 - Center: {ball_center_2}, Size: {ball_size_2}")


# Compute the distance in the second image using similar triangles
def compute_distance(distance_first ,ball_size_1,ball_size_2):
    return ( distance_first * ball_size_2 ) / ball_size_1


# Given real-world data
real_ball_size = 20  # in cm, assuming the real ball size is 20 cm
distance_first = 100  # The distance from the camera in the first image, in cm (1 meter = 100 cm)

# Calculate the distance to the ball in the second image
distance_second =compute_distance(distance_first ,ball_size_1,ball_size_2)

print(f"The distance to the ball in the second image is approximately {distance_second:.2f} cm")