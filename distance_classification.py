import cv2
import numpy as np

# Load images
img1 = cv2.imread("C:\Users\MEHER SIDHU\Downloads\Dr_Shashi_Tharoor.jpg")  # Change to actual file name
img2 = cv2.imread("C:\Users\MEHER SIDHU\Downloads\Plaksha_Faculty.jpg")

# Convert to grayscale (optional)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Convert images to arrays
img1_array = np.array(img1_gray)
img2_array = np.array(img2_gray)

print("Image 1 shape:", img1_array.shape)
print("Image 2 shape:", img2_array.shape)
