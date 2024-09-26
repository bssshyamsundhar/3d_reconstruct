import cv2
import numpy as np
from PIL import Image

def remove_background(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to get a binary image
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask with the same size as the image
    mask = np.zeros_like(img, dtype=np.uint8)

    # Draw the contours on the mask
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Convert the mask to grayscale
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Create an alpha channel with the mask
    alpha = mask_gray.astype(np.uint8)

    # Split the image into its color channels
    b, g, r = cv2.split(img)

    # Merge the color channels with the alpha channel
    rgba = [b, g, r, alpha]
    img_with_alpha = cv2.merge(rgba, 4)

    # Save the image with alpha channel
    final_img = np.array(img_with_alpha)
    return final_img
    # cv2.imwrite(output_path, img_with_alpha)

# Example usage
# remove_background('000004.png', 'output_image.png')
