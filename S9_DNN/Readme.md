### Albumentations, GradCam 
#### 1.	Apply tranformations using library Albumentations. Apply ToTensor, HorizontalFlip, Normalize (at min) + More (for additional points)
#### 2.	Please make sure that your test_transforms are simple and only using ToTensor and Normalize
#### 3.	Implement GradCam function as a module.
#### 4.	Your final code (notebook file) must use imported functions to implement transformations and GradCam functionality
#### 5.	Target Accuracy is 87%

### Results:
#### Arranging file structure

-----|models </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---  __init.py__ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- *Resnet* Files , resnetbase <br>
------|dataloader.py <br>
------|datamodel.py </br>
------|gradcam.py <br>
------|transformation.py <br>

#### After applying HorizontalFlip , Rotate, CoarseDropout, Normalize, ToTensor the accuracy is 88.40% at 24th epoch

### Loss Change Curve
<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S9_DNN/images/val_loss%20graph_change.png" width="550" title="Val Loss">

### Validation Accuracy Curve
<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S9_DNN/images/val_validation%20accuracy_change.png" width="550" title="Val Accuracy">

### Misclassified Images
<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S9_DNN/images/misclassified.png" width="550" title="Misclassified Img">

### Gradcam Images
<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S9_DNN/images/gradcam_incorrect_0_plane.png" width="550" title="Gradcam">

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S9_DNN/images/gradcam_incorrect_1_cat.png" width="550" title="Gradcam">

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S9_DNN/images/gradcam_incorrect_2_truck.png" width="550" title="Gradcam">

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S9_DNN/images/gradcam_incorrect_3_ship.png" width="550" title="Gradcam">

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S9_DNN/images/gradcam_incorrect_4_cat.png" width="550" title="Gradcam">
 
### Quiz S9
#### Training Accuracy 79.44%
#### Test Accuracy 81.14%%
#### Number of Epochs - 10
