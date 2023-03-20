import cv2

# TODO Implement the face detection in videostream using pre-recorded video with faces.
# TODO Implement the face detection in live videostream using web-camera.

# Face Detection using Viola-Jones Approach
img_path = "pic_group2.jpg"
Img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)

detector = cv2.CascadeClassifier()
cascadePath = 'haarcascade_frontalface_default.xml'
cascade_fn = cv2.samples.findFile(cascadePath)
detector.load(cascade_fn)


faces = detector.detectMultiScale(img_gray, scaleFactor=1.07, minNeighbors=3)

# Display found faces
img_out = Img.copy()
for (x, y, w, h) in faces:
    img_out = cv2.rectangle(img_out, (x, y, w, h), (0, 255, 255), 1)

# Load eyes cascade
eye_detector = cv2.CascadeClassifier()
cascade_fn = cv2.samples.findFile("haarcascade_eye.xml")
eye_detector.load(cascade_fn)
# For each face use it as a ROI  and detect eyes
for (x, y, w, h) in faces:
    img_faces = Img[y: y + h, x: x + w]
    eyes = eye_detector.detectMultiScale(img_faces, scaleFactor=1.05)
    for (x2, y2, w2, h2) in eyes:
        img_out = cv2.rectangle(img_out, (x + x2, y + y2, w2, h2), color=(147, 20, 255))

    # Define ROI and top 2/3 of the face image
    img_face_top = img_gray[y: y + h * 2 // 3, x: x + w]

cv2.imshow("Result", img_out)

cv2.waitKey(0)
