import cv2
import numpy as np

def main():
	# Load images
	try:
		img1 = cv2.imread(r"Dr_Shashi_Tharoor.jpg")  # Change to actual file name
		img2 = cv2.imread(r"Plaksha_Faculty.jpg")
	except Exception:
		return False

	if img1 == None or img2 == None:
		return False

	# Convert to grayscale (optional)
	img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	# Convert images to arrays
	img1_array = np.array(img1_gray)
	img2_array = np.array(img2_gray)

	return True

print(main())

