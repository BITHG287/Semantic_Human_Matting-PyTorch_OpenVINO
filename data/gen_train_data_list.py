import os
from shutil import copyfile


alpha_dir = "alpha"

image_dir = "image"
mask_dir = "mask"
trimap_dir = "trimap"


if __name__ == "__main__":

    train_txt = open("train.txt", "w")  

    for f in os.listdir(alpha_dir):
        tmp1, tmp2 = f.split(".")[0].split("-")
        search_dir = os.path.join("/media/hg/0009510B000EB710/dataset/semantic_human_dataset/clip_img", tmp1)
        
        for f1 in os.listdir(search_dir):
            for f2 in os.listdir(os.path.join(search_dir, f1)):
                if f2[:-4] == f[:-4]:
                    src_image = os.path.join(search_dir, f1, f2)
                    src_mask = src_image.replace("clip_img", "matting").replace("clip", "matting").replace(".jpg", ".png")

                    if os.path.exists(src_image) and os.path.exists(src_mask) and src_image.endswith(".jpg"):
                        train_txt.write(f[:-4] + "\n")
                        
                        dst_image = os.path.join(image_dir, f2)
                        dst_mask = dst_image.replace(image_dir, mask_dir).replace(".jpg", ".png")
                        # copyfile(src_image, dst_image)
                        # copyfile(src_mask, dst_mask)

                        print(f)
                        print("    ", dst_image)
                        print("    ", dst_mask)
        
    train_txt.close()
