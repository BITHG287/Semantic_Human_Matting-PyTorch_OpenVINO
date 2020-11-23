import time
import cv2
import torch 
import argparse
import numpy as np
import os
import torch.nn.functional as F
torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description='human matting')
parser.add_argument('--model', default='./ckpt/human_matting/model/model_obj.pth', help='preTrained model')
parser.add_argument('--size', type=int, default=256, help='input size')
parser.add_argument('--without_gpu', action='store_true', default=False, help='no use gpu')

args = parser.parse_args()
torch.set_grad_enabled(False)

    
#################################
#----------------
if args.without_gpu:
    print("use CPU !")
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print("----------------------------------------------------------")
        print("|       use GPU !      ||   Available GPU number is {} !  |".format(n_gpu))
        print("----------------------------------------------------------")

        device = torch.device('cuda:0,1')

#################################
#---------------
def load_model(args):
    print('Loading model from {}...'.format(args.model))
    if args.without_gpu:
        myModel = torch.load(args.model, map_location=lambda storage, loc: storage)
    else:
        myModel = torch.load(args.model)

    myModel.eval()
    myModel.to(device)
    
    return myModel


def seg_process(args, image, net):

    origin_h, origin_w, c = image.shape
    image_resize = cv2.resize(image, (args.size,args.size), cv2.INTER_CUBIC)
    # print(image_resize.shape)

    image_resize = (image_resize - (104., 112., 121.,)) / 255.0
    tensor_4D = torch.FloatTensor(1, 3, args.size, args.size)
    tensor_4D[0,:,:,:] = torch.FloatTensor(image_resize.transpose(2,0,1))
    inputs = tensor_4D.to(device)

    t0 = time.time()
    trimap, alpha = net(inputs)
    print('infer cost time:', (time.time() - t0))  

    if args.without_gpu:
        alpha_np = alpha[0,0,:,:].data.numpy()
        trimap_np = trimap[0,0,:,:].data.numpy()
    else:
        alpha_np = alpha[0,0,:,:].cpu().data.numpy()
        trimap_np = trimap[0,0,:,:].cpu().data.numpy()

    alpha_np = cv2.resize(alpha_np, (origin_w, origin_h), cv2.INTER_CUBIC)
    fg = np.multiply(alpha_np[..., np.newaxis].astype(np.float32), image.astype(np.float32))

    bg = image
    bg_gray = np.multiply(1-alpha_np[..., np.newaxis], image)
    bg_gray = cv2.cvtColor(bg_gray, cv2.COLOR_BGR2GRAY)

    bg[:,:,0] = bg_gray
    bg[:,:,1] = bg_gray
    bg[:,:,2] = bg_gray

    out = fg + bg
    out[out<0] = 0
    out[out>255] = 255
    out = out.astype(np.uint8)

    return out


def camera_seg(args, net):
    cv2.namedWindow("img", 0)
    cv2.namedWindow("img_seg", 0)
    for f in os.listdir("./images"):
        img_path = os.path.join("./images", f)

        img = cv2.imread(img_path)
        img_seg = seg_process(args, img, net)
        # cv2.imwrite("img_seg.png", img_seg)

        cv2.imshow("img", img)
        cv2.imshow("img_seg", img_seg)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()


def main(args):
    myModel = load_model(args)
    camera_seg(args, myModel)

if __name__ == "__main__":
    main(args)
