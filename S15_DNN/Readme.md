
### Depth Estimation and Segmentation 

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S15_DNN/images/asgmnt.PNG" width="750" title="Que"> 

####  Create a custom dataset for monocular depth estimation and segmentation simultaneously. Since we do not have access to a depth camera, we use a pre-trained depth model to ####  generate the depth maps which will be used as the ground truth for our model.

### Dataset Creation


1.	Download Background images from internet, from google images
2.	Download transparent Foreground images from internet, from google images
3.	Resize background & foreground images <br/>
https://github.com/haricharanvihari/extensive_viz/blob/master/S15_DNN/utils/image_resize.py

4.	Generate Mask images for all the foreground images - Code <br/>
https://github.com/haricharanvihari/extensive_viz/blob/master/S15_DNN/utils/mask.py

5.	Generate Overlay images by putting foreground images over background images - Code & Colab <br/>
https://github.com/haricharanvihari/extensive_viz/blob/master/S15_DNN/utils/overlay_mask_creation.ipynb

6.	Generate mask images for overlay images - Code <br/>
https://github.com/haricharanvihari/extensive_viz/blob/master/S15_DNN/utils/overlay.py

7.	Generate Depth maps for overlay images - Colab <br/>
https://github.com/haricharanvihari/extensive_viz/blob/master/S15_DNN/Dense_depth.ipynb

8.	Calculate Images dimensions - Colab <br/>
https://github.com/haricharanvihari/extensive_viz/blob/master/S15_DNN/Dataset.ipynb

Dataset Link <br/>
Link to dataset - https://drive.google.com/drive/folders/16PIIckM14zXzH1KQxH3vGfiv7Ia-_4wA?usp=sharing
Dataset Samples

#### Background (bg) <br/>
•	99 Images of places like restaurent, malls or home.<br/>
•	Each image was resized to 224 x 224. Final Image dimensions: (224, 224, 3). Total size: 1.5M<br/>
•	Mean: [0.5442, 0.5057, 0.4621]<br/>
•	Std: [0.2609, 0.2624, 0.2799]<br/>
<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S15_DNN/images/bg.PNG" width="750" title="Que"> <br/>

#### Foreground (fg)<br/>
•	99 Images of humans & very few animals with transparent background.<br/>
•	Images were rescaled to keep height 140 and resizing width while maintaining aspect ratio. Code to resize images<br/>
•	Image dimensions: (140, width, 4)<br/>
•	Directory size: 1.8M<br/>

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S15_DNN/images/fg.PNG" width="750" title="Que"> <br/>

#### Foreground Mask (fg_mask)<br/>
•	For every foreground, corresponding mask image was created<br/>
•	Wrote custom code to generate foreground mask<br/>
•	Image dimensions: (140, width)<br/>
•	Directory size: 848k<br/>
<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S15_DNN/images/fg_mask.PNG" width="750" title="Que"> <br/>

#### Foreground Overlayed on Background (bgfg)<br/>
•	Overlay each foreground image on each background randomly 20 times + 20 times more with fliping the foreground<br/>
•	Colab File for generating overlay images<br/>
•	Code to generate overlay images<br/>
•	Number of images: 99 * 99 * 2 * 20 = 3,92,040<br/>
•	Image dimensions: (224, 224, 3)<br/>
•	Directory size: 5.3G<br/>
•	Mean: [0.5365, 0.4978, 0.4601]<br/>
•	Std: [0.2671, 0.2650, 0.2799]<br/>

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S15_DNN/images/bgfg.PNG" width="750" title="Que"> <br/>

#### Foreground Overlayed on Background Mask (bgfg_mask)<br/>
•	For every overlayed image, corresponding mask image was created<br/>
•	Wrote custom code to generate mask<br/>
•	Number of images: 3,92,041<br/>
•	Image dimensions: (224, 224)<br/>
•	Directory size: 1.6G<br/>
•	Mean: [0.0943]<br/>
•	Std: [0.2856]<br/>

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S15_DNN/images/bgfg_mask.PNG" width="750" title="Que"> <br/>

#### Foreground Overlayed on Background Depth Map (bgfg_depth)<br/>
•	Depth map was generated for every overlay image.<br/>
•	A pre-trained monocular depth estimation model DenseDepth was used to generate the depth maps.<br/>
•	Image was stored as a grayscale image.<br/>
•	Number of images: 3,92,041<br/>
•	Image dimensions: (224, 224)<br/>
•	Directory size: 1.6G<br/>
•	Mean: [0.4334]<br/>
•	Std: [0.2715]<br/>
                   
<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S15_DNN/images/bgfg_depth.PNG" width="750" title="Que"> <br/>
