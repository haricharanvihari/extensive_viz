
### S11 Super-Convergence
#### 1.	Write a code that draws zigzag curve
#### 2.	Write a code which uses this new ResNet Architecture for Cifar10: a. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k] b. Layer1 -
####  o	X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
####  o	R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
####  o	Add(X, R1) c. Layer 2 -
####  o	Conv 3x3 [256k]
####  o	MaxPooling2D
####  o	BN
####  o	ReLU d. Layer 3 -
####  o	X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
####  o	R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
####  o	Add(X, R2) e. MaxPooling with Kernel Size 4 f. FC Layer g. SoftMax
#### 3.	Uses One Cycle Policy such that:-
####  o	Total Epochs = 24
####  o	Max at Epoch = 5
####  o	LRMIN = FIND
####  o	LRMAX = FIND
####  o	NO Annihilation
#### 4.	Uses this transform - RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
#### 5.	Batch size = 512
#### 6.	Target Accuracy: 90%.



### Results:
#### Arranging file structure

-----|models </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---  __init.py__ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- *Resnet* Files , resnetbase <br>
------|dataloader.py <br>
------|datamodel.py </br>
------|gradcam.py <br>
------|lrfinder.py  <br>
------|cycliclr.py  <br>
------|optimizer.py <br>
------|transformation.py <br>

#### 1.	Added Dropout in Albumentation
#### 2.	Implemented LR Finder for SGD taken from Source: https://github.com/davidtvs/pytorch-lr-finder
#### 3.	Implemented ReduceLROnPlatea. Source: https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
#### 4.	Best Learning Rate - 0.001
#### 5.	Accuracies in Last few epochs ranging Train 97.4%, Test: 88.5%
#### 6. Best Accuracy Train - 97.58%(Epoch 50) Test - 88.83%(Epoch 50)

### 1.	Cyclic Curve
<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/images/CYCLIC_TRIANGLE.png.png" width="550" title="Cyclic">

####  o	LR Max - 0.07
####  o	Best Accuracy Train - 94.26% Test - 90.46% (Epoch 5)
####  o	Target Accuracy : 88.51%

### 2.	Train Test Acc Curve
<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/images/train_test_acc_change.png" width="550" title="Train_test">

### 3.	Loss Acc Change Curve
<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/images/loss_acc_change.png" width="550" title="Train_test">

### Gradcam Images
<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/Images/gradcam_incorrect_0_dog.png" width="550" title="car">

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/Images/gradcam_incorrect_1_plane.png" width="550" title="truck">

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/Images/gradcam_incorrect_2_cat.png" width="550" title="cat">

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/Images/gradcam_incorrect_3_ship.png" width="550" title="dog">

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/Images/gradcam_incorrect_4_dog.png" width="550" title="dog">

