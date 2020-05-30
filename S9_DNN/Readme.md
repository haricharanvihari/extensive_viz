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
<img width=“600” alt=“img1” src=“https://github.com/haricharanvihari/extensive_viz/blob/master/S9_DNN/images/val_loss%20graph_change.png”>
