import cv2
import numpy as np

# TODO Create class and methods

# Features Detectors
img_path = "pic_totoro.jpg"

# SIFT Detector
input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Image reading
Img = cv2.resize(input_img, (1000, 600))
img_gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)  # Convert from RGB to grayscale

# Creating an instance of class SIFT and limiting it to detect only x the strongest features
feature_points = 100
sift = cv2.SIFT_create(feature_points)
Ifp = sift.detect(img_gray)

# Displaying the feature points. Only feature point position is displayed by default
# img_out = cv2.drawKeypoints(Img, Ifp, None)
# Displaying SIFT feature points in green color with scale and orientation
img_out = cv2.drawKeypoints(Img, Ifp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("SIFT detector", img_out)

# ORB Detection
input_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
Img = cv2.resize(input_img, (1000, 600))
img_gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(100)
Ifp = orb.detect(img_gray)
img_out = cv2.drawKeypoints(Img, Ifp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("ORB detection", img_out)

# Feature point description method
# starts with SIFT detection
img1 = cv2.imread("pic1.jpg", cv2.IMREAD_COLOR)
img2 = cv2.imread("pic2.jpg", cv2.IMREAD_COLOR)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
img1_fp, img1_des = sift.detectAndCompute(img1_gray, None)
img2_fp, img2_des = sift.detectAndCompute(img2_gray, None)

# Matching feature points descriptors on one image with descriptors on the second picture
# Creating brute force descriptor matcher. BFMatcher works good for SIFT descriptor
matcher_bf = cv2.BFMatcher(crossCheck=False)

# Brute force descriptor matcher with Hamming distance
matcher_bf_dist = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# FLANN 5 KD-trees descriptor matcher foe SIFT descriptors
FLANN_INDEX_KDTREE = 1
index_params_kd = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
matcher_flann = cv2.FlannBasedMatcher(index_params_kd, dict())

# FLANN LSH descriptor matcher for ORB descriptors
FLANN_INDEX_LSH = 6
index_params_lsh = dict(algorithm=FLANN_INDEX_LSH, table_number=6,
                        key_size=12, multi_probe_level=1)
matcher_lsh = cv2.FlannBasedMatcher(index_params_lsh, dict())

# Finding single best match for two sets of descriptors
# matches = matcher_bf.match(img1_des, img2_des)

# Finding k-nearest best match for two sets of descriptors and filtering them
# Find KNN matches with k = 2
matches = matcher_bf.knnMatch(img1_des, img2_des, k=2)
# Select good matches
knn_ration = 0.75
good = []
for m in matches:
    if len(m) > 1:
        if m[0].distance < knn_ration * m[1].distance:
            good.append(m[0])
matches = good

# Displaying top 10 matches
num_matches = 10
matches = sorted(matches, key=lambda x: x.distance)
img_match = cv2.drawMatches(img1, img1_fp, img2, img2_fp, matches[: num_matches], None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0, 255, 0))

# Executing RANSAC to calculate the transformation matrix
MIN_MATCH_COUNT = 10
if len(matches) < MIN_MATCH_COUNT:
    print(" Not␣enough␣matches.")
    # return
# Create arrays of point coordinates
img1_pts = np.float32([img1_fp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
img2_pts = np.float32([img2_fp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
# Run RANSAC method
M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5)
mask = mask.ravel().tolist()

# Displaying the location of the first image on the second one
# Image corners
h, w = img1.shape[:2]
img1_box = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
img1_to_2box = cv2.perspectiveTransform(img1_box, M)
# Draw a red box on the second image
img2_res = cv2.polylines(img2, [np.int32(img1_to_2box)], True, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow("Search result", img2_res)

# Displaying inlier matches
img_trans = cv2.drawMatches(img1, img1_fp, img2_res, img2_fp, matches, None,
                            matchesMask=mask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                            matchColor=(0, 255, 0))
cv2.imshow("Transformation", img_trans)


cv2.waitKey(0)
