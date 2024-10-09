import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def apply_radial_distortion(image, k1, k2, k3):
    h, w = image.shape[:2]
    # Generate the camera matrix
    fx = w  # focal length in x
    fy = h  # focal length in y
    cx = w / 2  # principal point x
    cy = h / 2  # principal point y
    
    camera_matrix = np.array([[fx, 0, cx],
                               [0, fy, cy],
                               [0, 0, 1]], dtype=np.float32)
    
    # Initialize the distortion coefficients
    dist_coeffs = np.array([k1, k2, k3, 0, 0], dtype=np.float32)
    
    # Generate new camera matrix based on the distortion
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0)
    
    # Apply distortion
    distorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    return distorted_image

# Load your selfie image
selfie_image = cv2.imread('../origin.jpg')

# Apply barrel distortion (positive coefficients)
barrel_distorted_image = apply_radial_distortion(selfie_image, k1=0.7, k2=0.2, k3=0.0)

# Apply pincushion distortion (negative coefficients)
pincushion_distorted_image = apply_radial_distortion(selfie_image, k1=-0.7, k2=0.2, k3=0.0)

# Save the results to the 'result' directory
cv2.imwrite('../results/ex2_barrel.jpg', barrel_distorted_image)
cv2.imwrite('../results/ex2_pincushion.jpg', pincushion_distorted_image)

# Display the results
plt.figure(figsize=(12, 8))

# Show original image
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(selfie_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Show barrel distorted image
plt.subplot(1, 3, 2)
plt.title('Barrel Distortion')
plt.imshow(cv2.cvtColor(barrel_distorted_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Show pincushion distorted image
plt.subplot(1, 3, 3)
plt.title('Pincushion Distortion')
plt.imshow(cv2.cvtColor(pincushion_distorted_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
