
### Depth Estimation and Segmentation 

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/images/S11.png" width="750" title="Que"> 

####  Create a custom dataset for monocular depth estimation and segmentation simultaneously. Since we do not have access to a depth camera, we use a pre-trained depth model to ####  generate the depth maps which will be used as the ground truth for our model.

### Dataset Creation


1.	Download Background images from internet, from google images
2.	Download transparent Foreground images from internet, from google images
3.	Resize background & foreground images - Code
4.	Generate Mask images for all the foreground images - Code
5.	Generate Overlay images by putting foreground images over background images - Code & Colab
6.	Generate mask images for overlay images - Code
7.	Generate Depth maps for overlay images - Colab
8.	Calculate Images dimensions - Colab
Dataset Link
Link to dataset - https://drive.google.com/drive/folders/1IvKUbm4WcqNUS9dkOV_gPSfCtnmPIg8Y?usp=sharing
Dataset Samples
Background (bg)
•	99 Images of places like restaurent, malls or home.
•	Each image was resized to 224 x 224. Final Image dimensions: (224, 224, 3). Total size: 1.5M
•	Mean: [0.5442, 0.5057, 0.4621]
•	Std: [0.2609, 0.2624, 0.2799]
                   
Foreground (fg)
•	99 Images of humans & very few animals with transparent background.
•	Images were rescaled to keep height 140 and resizing width while maintaining aspect ratio. Code to resize images
•	Image dimensions: (140, width, 4)
•	Directory size: 1.8M
                   
Foreground Mask (fg_mask)
•	For every foreground, corresponding mask image was created
•	Wrote custom code to generate foreground mask
•	Image dimensions: (140, width)
•	Directory size: 848k
                   
Foreground Overlayed on Background (bgfg)
•	Overlay each foreground image on each background randomly 20 times + 20 times more with fliping the foreground
•	Colab File for generating overlay images
•	Code to generate overlay images
•	Number of images: 99 * 99 * 2 * 20 = 3,92,040
•	Image dimensions: (224, 224, 3)
•	Directory size: 5.3G
•	Mean: [0.5365, 0.4978, 0.4601]
•	Std: [0.2671, 0.2650, 0.2799]
                   
Foreground Overlayed on Background Mask (bgfg_mask)
•	For every overlayed image, corresponding mask image was created
•	Wrote custom code to generate mask
•	Number of images: 3,92,041
•	Image dimensions: (224, 224)
•	Directory size: 1.6G
•	Mean: [0.0943]
•	Std: [0.2856]
                   
Foreground Overlayed on Background Depth Map (bgfg_depth)
•	Depth map was generated for every overlay image.
•	A pre-trained monocular depth estimation model DenseDepth was used to generate the depth maps.
•	Image was stored as a grayscale image.
•	Number of images: 3,92,041
•	Image dimensions: (224, 224)
•	Directory size: 1.6G
•	Mean: [0.4334]
•	Std: [0.2715]
                   

