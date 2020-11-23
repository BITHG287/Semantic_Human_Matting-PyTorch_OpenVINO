# 0. None

The project refers to https://github.com/lizhengwei1992/Semantic_Human_Matting.

Finally, I converted the pytorch model to the openvino model, and run it on cpu.

# 1. My Environment: 

## for python:

python == 3.6.5

pytorch-gpu == 1.2.0

torchvision == 0.4.0

cv2 == 4.1.2

## for C++:
    
openvino == 2019.3.334

opencv == 4.1.2

# 2. Prepare train dataset

Download alpha images from: https://pan.baidu.com/s/1KfPgNQxpcxtWOnK-N6-TqA (t48e)

Download init images and mask images from: https://pan.baidu.com/s/1R9PJJRT-KjSxh-2-3wCGxQ (dzsn)

Unzip alpha to data folder and execute:

(1) cd data

(2) use alpha images to generate trimap images: python gen_trimap.py

(3) generate train.txt: python gen_train_data_list.py

# 3. Train

I set patch_size = 256, train_batch = 32.

(1) train phase 1:

python train.py --finetuning False --lr 1e-3 --nEpochs 100 --train_phase pre_train_t_net

(2) train phase 2:

python train.py --finetuning True --lr 1e-4 --nEpochs 200 --train_phase end_to_end

# 4. Test

Put images into images folder and run python test_image.py.

# 5. OpenVINO

(1) Convert model_obj.pth to model_obj.onnx: python torch_to_onnx.py

(2) Copy model_obj.onnx to C:\IntelSWTools\openvino_2019.3.334\deployment_tools\model_optimizer, and run:
```
python mo_onnx.py --input_shape=[1,3,256,256] --mean_values [104.0,112.0,121.0] --scale_values [255.0,255.0,255.0] --input_model .\model_obj.onnx -o .\
```
Alter that, we have two files: model_obj.xml(model information file) and model_obj.bin(weights file).

# 6. Qt

Build Kit: Desktop Qt 5.12.0 MSVC2015 64bit, Release.

OpenVINO model files: openvino_model/model_obj.xml, openvino_model/model_obj.bin.

(1) Create a new qt console project.

(2) Copy the content of file 'for_qt/console_test.pro' to your own .pro file.

(3) Use file 'for_qt/main.cpp' in your project.
