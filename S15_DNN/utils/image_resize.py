import os
from PIL import Image

from resizeimage import resizeimage

filename = "1234.png"
#base_folder = os.path.join("C:\\", "Anurag", "Personal", "ML", "EVA", "github", "fg_bg_dataset")
base_folder = 'C:/Users/nalluru.vihari/Desktop/EVA/nts/S15A_DNN/fg_dwnd/'
# base_folder = "C:\Anurag\Personal\ML\EVA\github\fg_bg_dataset"

input_image_folder = 'C:/Users/nalluru.vihari/Desktop/EVA/nts/S15A_DNN/fg_dwnd/'
# input_image_folder = os.path.join(base_folder, "fg_original")
#output_image_folder = os.path.join(base_folder, "fg")

output_image_folder = 'C:/Users/nalluru.vihari/Desktop/EVA/nts/S15A_DNN/fg/'

for count, filename in enumerate(os.listdir(input_image_folder)):
    with open(os.path.join(input_image_folder, filename), 'r+b') as f:
        with Image.open(f) as image:
            temp = None
            width, height = image.size
            if width < height:
                temp = height
            else:
                temp = width
            
            ratio = temp / 140
            new_width, new_height = width/ratio, height/ratio
            # print(int(new_width), int(new_height))
            cover = resizeimage.resize_cover(image, [int(new_width), int(new_height)])
            cover.save(os.path.join(output_image_folder,filename), image.format)