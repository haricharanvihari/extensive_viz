import io
import os
import numpy as np
from PIL import Image

from tqdm import tqdm

flip_methods = [Image.FLIP_TOP_BOTTOM, Image.FLIP_LEFT_RIGHT, Image.ROTATE_90,
                  Image.ROTATE_180, Image.ROTATE_270, Image.TRANSPOSE, Image.TRANSVERSE]

def overlay_fg_on_bg(base_input_folder, base_output_folder):
  # fg_img_folder = os.path.join(base_input_folder, "fg")
  fg_img_folder = 'C:/Users/nalluru.vihari/Desktop/EVA/nts/S15A_DNN/fg/'
  fg_mask_folder = 'C:/Users/nalluru.vihari/Desktop/EVA/nts/S15A_DNN/fg_mask/'
  bg_img_folder = 'C:/Users/nalluru.vihari/Desktop/EVA/nts/S15A_DNN/bg/'
  bgfg_overlay_folder = 'C:/Users/nalluru.vihari/Desktop/EVA/nts/S15A_DNN/fg/bgfg_overlay/'
  bgfg_mask_folder = 'C:/Users/nalluru.vihari/Desktop/EVA/nts/S15A_DNN/fg/bgfg_overlay/bgfg_mask'

  bg_imgs = sorted(os.listdir(bg_img_folder))
  fg_imgs = sorted(os.listdir(fg_img_folder))
  
  pbar = tqdm(enumerate(bg_imgs))

  for bidx, bg_img in pbar:
    pbar.set_description("Processing %s" % bg_img)
    bg_num = bg_img.split('.')[0] #.split('_')[1]
    with Image.open(os.path.join(bg_img_folder, bg_img)) as mbg:
      for fidx, fg_img in enumerate(fg_imgs):
        pbar.set_description("Processing %s for FG %s" % (bg_img, fg_img))
        fg_num = fg_img.split('.')[0] #.split('_')[1]
        with Image.open(os.path.join(fg_img_folder, fg_img)).convert("RGBA") as mfg, \
          Image.open(os.path.join(fg_mask_folder, "mask_" + fg_img)) as mfg_mask:

          for i in range(20):
            for should_flip in [True, False]:
              flag = ""
              bg = mbg.copy()
              fg = mfg.copy()
              fg_mask = mfg_mask.copy()

              if should_flip == True:
                flip_m_len = len(flip_methods)
                pick_index = i
                if i >= flip_m_len:
                  pick_index = i % flip_m_len

                flag = "f"
                
                flip_method = flip_methods[pick_index]
                fg = fg.transpose(flip_method)
                fg_mask = fg_mask.transpose(flip_method)

              bg_w, bg_h = bg.size
              fg_w, fg_h = fg.size
              max_h = bg_h - fg_h
              max_w = bg_w - fg_w
              pos_x = np.random.randint(low=0, high=max_w, size=1)[0]
              pos_y = np.random.randint(low=0, high=max_h, size=1)[0]

              bg.paste(fg, (pos_x, pos_y), fg)

              bg_mask = Image.new('L', bg.size)
              fg_mask = fg_mask.convert('L')
              bg_mask.paste(fg_mask, (pos_x, pos_y), fg_mask)

              img_substring = "ol_" + "bg" + bg_num + "fg" + fg_num + str(i + 1) + flag + "_" + fg_img
              bg.save(os.path.join(bgfg_overlay_folder, img_substring), optimize=True, quality=65, format="JPEG")
              bg_mask.save(os.path.join(bgfg_mask_folder, "mask_" + img_substring), optimize=True, quality=65, format="JPEG")
              
              del fg
              del bg
    #if bidx == 0:
    #  break

if __name__ == '__main__':
    # input_folder = os.path.join("C:\\", "Anurag", "Personal", "ML", "EVA", "github", "fg_bg_dataset")
    # output_folder = os.path.join("C:\\", "Anurag", "Personal", "ML", "EVA", "github", "fg_bg_dataset")

    input_folder = 'C:/Users/nalluru.vihari/Desktop/EVA/nts/S15A_DNN/'
    output_folder = 'C:/Users/nalluru.vihari/Desktop/EVA/nts/S15A_DNN/'
    overlay_fg_on_bg(input_folder, output_folder)