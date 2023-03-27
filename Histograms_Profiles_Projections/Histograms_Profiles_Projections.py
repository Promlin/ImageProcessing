import cv2

# Image Histogram
# I is an RGB -image
# Number of histogram bins
histSize = 256

# Histogram range
# The upper boundary is exclusive
histRange = (0, 256)

img_path = "stray.jpg"
Img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# Split an image into color layers
# OpenCV stores RGB image as BGR
I_BGR = cv2.split(Img)
# Calculate a histogram for each layer
bHist = cv2.calcHist(I_BGR, [0], None, [histSize], histRange)
gHist = cv2.calcHist(I_BGR, [1], None, [histSize], histRange)
rHist = cv2.calcHist(I_BGR, [2], None, [histSize], histRange)

# Arithmetic Operations

cv2.imshow("Result", bHist)

cv2.waitKey(0)
