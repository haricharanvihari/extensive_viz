import os
from PIL import Image
from PIL.ImageColor import getcolor, getrgb
from PIL.ImageOps import grayscale

# base_folder = os.path.join("C:\\", "Anurag", "Personal", "ML", "EVA", "github", "fg_bg_dataset")
base_folder = 'C:/Users/nalluru.vihari/Desktop/EVA/nts/S15A_DNN/'

def image_tint(src, tint=None):
    src1 = Image.open(src)
    
    src = src1.convert('RGBA')
    if src.mode not in ['RGB', 'RGBA']:
        raise TypeError('Unsupported source image mode: {}'.format(src.mode))
    src.load()

    tr, tg, tb = 255, 255, 255
    if tint:
      tl = getcolor(tint, "L")  # tint color's overall luminosity
    else:
      tl = 1
    if not tl: tl = 1  # avoid division by zero
    tl = float(tl)  # compute luminosity preserving tint factors
    sr, sg, sb = map(lambda tv: tv/tl, (tr, tg, tb))  # per component adjustments

    # create look-up tables to map luminosity to adjusted tint
    # (using floating-point math only to compute table)
    luts = (list(map(lambda lr: int(lr*sr + 0.5), range(256))) +
            list(map(lambda lg: int(lg*sg + 0.5), range(256))) +
            list(map(lambda lb: int(lb*sb + 0.5), range(256))))
    
    l = grayscale(src)  # 8-bit luminosity version of whole image
    if Image.getmodebands(src.mode) < 4:
        merge_args = (src.mode, (l, l, l))  # for RGB verion of grayscale
    else:  # include copy of src image's alpha layer
        a = Image.new("L", src.size)
        a.putdata(src.getdata(3))
        merge_args = (src.mode, (l, l, l, a))  # for RGBA verion of grayscale
        luts += range(256)  # for 1:1 mapping of copied alpha values

    return Image.merge(*merge_args).point(luts)

if __name__ == '__main__':

    # fg_img_folder = os.path.join(base_folder, "fg")
    # fg_mask_folder = os.path.join(base_folder, "fg_mask")

    fg_img_folder = 'C:/Users/nalluru.vihari/Desktop/EVA/nts/S15A_DNN/fg/'
    fg_mask_folder = 'C:/Users/nalluru.vihari/Desktop/EVA/nts/S15A_DNN/fg_mask/'

    for count, filename in enumerate(os.listdir(fg_img_folder)):
        input_image_path = os.path.join(fg_img_folder, filename)
        print ('masking "{} image: {}"'.format(count, input_image_path))
        
        temp_mask_file = os.path.join(fg_mask_folder, "temp_mask_" + filename)
        black_image_file = os.path.join(fg_mask_folder, "black_" + filename)
        mask_file_name = os.path.join(fg_mask_folder, "mask_" + filename)
        
        result = image_tint(input_image_path)
        result.save(temp_mask_file)
        image_size = result.size
        result.close()
    
        # Create Black Image
        black_image = Image.new('RGB', (image_size))
        black_image.save(black_image_file, "PNG")
        black_image.close()
        
        temp_mask_image = Image.open(temp_mask_file)
        black_image = Image.open(black_image_file)

        # Creating New image - adding black image then original image as mask
        mask_image = Image.new('RGBA', (image_size), (0, 0, 0, 0))
        mask_image.paste(black_image, (0,0))
        mask_image.paste(temp_mask_image, (0,0), mask=temp_mask_image)
        mask_image.save(mask_file_name, format="png")
        mask_image.close()

        temp_mask_image.close()
        black_image.close()
        
        os.remove(temp_mask_file)
        os.remove(black_image_file)
    print ('done')