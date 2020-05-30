### Albumentations, GradCam 
#### 1.	Apply tranformations using library Albumentations. Apply ToTensor, HorizontalFlip, Normalize (at min) + More (for additional points)
#### 2.	Please make sure that your test_transforms are simple and only using ToTensor and Normalize
#### 3.	Implement GradCam function as a module.
#### 4.	Your final code (notebook file) must use imported functions to implement transformations and GradCam functionality
#### 5.	Target Accuracy is 87%

### Results:
#### Arranging file structure

-----|models
	|---  __init.py__
	|--- *Resnet* Files , resnetbase
------|dataloader.py
------|datamodel.py
------|gradcam.py
------|transformation.py

#### After applying HorizontalFlip , Rotate, CoarseDropout, Normalize, ToTensor the accuracy is 88.40% at 24th epoch
