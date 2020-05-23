## Objective:
1.	To change the code to make use of GPU
2.	Change the architecture to C1C2C3C40 (basically 3 MPs)
3.	Total RF must be more than 44
4.	One of the layers must use Depthwise Separable Convolution
5.	One of the layers must use Dilated Convolution
6.	Use GAP (compulsory):- add FC after GAP to target #of classes (optional)
7.	Achieve 80% accuracy, as many epochs as you want. Total Params to be less than 1M

## Results:
1.  Changed the code such that it uses GPU
2.  Updated the architecture to C1C2C3C4
3.  Total RF > than 44
4.  Used Depthwise Separable Convolution , Dilated Convolution
5.  Used GAP.
6.  Highest Accuracy: 82.71% in 19th Epoch
7.  Accuracy in 20th Epoch : 82.58%
8.  Total Params: 76,384
