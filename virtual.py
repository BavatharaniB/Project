import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the original image and background
input_image = cv2.imread('aunty.png')
background_image = cv2.imread('background.jpg')

# Resize the background to match input image
background_image = cv2.resize(background_image, (input_image.shape[1], input_image.shape[0]))

# Convert input image to HSV color space
hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

# Define HSV range for green screen
lower_bound = np.array([35, 40, 40])
upper_bound = np.array([85, 255, 255])

# Create masks
mask = cv2.inRange(hsv, lower_bound, upper_bound)
mask_inv = cv2.bitwise_not(mask)

# Extract the foreground and background
foreground = cv2.bitwise_and(input_image, input_image, mask=mask_inv)
background = cv2.bitwise_and(background_image, background_image, mask=mask)

# Combine both
result = cv2.add(foreground, background)

# Save the result
cv2.imwrite('output.jpg', result)

# Show result (optional if you want to visualize it)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Output Image')
plt.show()