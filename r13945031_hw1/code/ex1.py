import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to scale image using bilinear and bicubic interpolation
def scale_image(image, scale_factor, interpolation_method):
    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=interpolation_method)
    return scaled_image

# Load the selfie image (ensure the size is at least 512x512 pixels)
image = cv2.imread('../origin.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display in matplotlib

# Scale factors
shrink_factor = 0.12
zoom_factor = 7

# Step 1: Shrink the image
bilinear_shrunk = scale_image(image, shrink_factor, cv2.INTER_LINEAR)
bicubic_shrunk = scale_image(image, shrink_factor, cv2.INTER_CUBIC)

# Step 2: Zoom the shrunk image
bilinear_zoomed = scale_image(bilinear_shrunk, zoom_factor, cv2.INTER_LINEAR)
bicubic_zoomed = scale_image(bicubic_shrunk, zoom_factor, cv2.INTER_CUBIC)

# Save the results to the 'result' directory
cv2.imwrite('../results/ex1_bilinear_shrink.jpg', cv2.cvtColor(bilinear_shrunk, cv2.COLOR_RGB2BGR))
cv2.imwrite('../results/ex1_bicubic_shrink.jpg', cv2.cvtColor(bicubic_shrunk, cv2.COLOR_RGB2BGR))
cv2.imwrite('../results/ex1_bilinear_zoom.jpg', cv2.cvtColor(bilinear_zoomed, cv2.COLOR_RGB2BGR))
cv2.imwrite('../results/ex1_bicubic_zoom.jpg', cv2.cvtColor(bicubic_zoomed, cv2.COLOR_RGB2BGR))

# Display the results
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].imshow(bilinear_shrunk)
ax[0, 0].set_title('Shrunk - Bilinear Interpolation')

ax[0, 1].imshow(bicubic_shrunk)
ax[0, 1].set_title('Shrunk - Bicubic Interpolation')

ax[1, 0].imshow(bilinear_zoomed)
ax[1, 0].set_title('Zoomed - Bilinear Interpolation')

ax[1, 1].imshow(bicubic_zoomed)
ax[1, 1].set_title('Zoomed - Bicubic Interpolation')

plt.show()
