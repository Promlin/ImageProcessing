# Face Detection

In this project I tried to detect faces and part of the face (eyes and noses) by using Viola-Jones approach.

The explanation of the principle is quite voluminous, so I decided to attach links to Russian-language and English-language articles, where the theory is explained quite well

[Russian - "Метод Виолы-Джонса (Viola-Jones) как основа для распознавания лиц"](https://habr.com/ru/post/133826/)

[English - "Viola Jones Algorithm and Haar Cascade Classifier"](https://towardsdatascience.com/viola-jones-algorithm-and-haar-cascade-classifier-ee3bfb19f7d8)

On the following picture you can see detection with the only one face on the picture. There are some mistakes with the face detection (there are two false detected faces), but the face itself is correctly defined and the eyes with the nose are also correctly defined. 
Image (a) is the original image, and image (b) shows the result of the method.

![image](https://user-images.githubusercontent.com/48473061/227621847-782dcb80-f05e-4037-ad48-06e57627bcdd.png)

The image below shows face detection using the example of a photo with several people. The detection was performed almost without errors (except for the duplication of one of the faces). 
Image (a) is the original image, and image (b) shows the result of the method.

![image](https://user-images.githubusercontent.com/48473061/227622034-0cbb1d59-a7ef-4b38-a404-16a9e55484ff.png)
