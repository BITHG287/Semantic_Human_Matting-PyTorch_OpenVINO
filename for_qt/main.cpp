
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <vector>

#include "inference_engine.hpp"
#include "samples/ocv_common.hpp"
#include "ext_list.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
using namespace std;
using namespace cv;
using namespace InferenceEngine;


std::string DEVICE = "CPU";
std::string imageFile = "D:\\1.jpg";
std::string binFile = "D:\\model_obj.bin";
std::string xmlFile = "D:\\model_obj.xml";


int main(int argc, char *argv[])
{
    // --------------------------- 1. 载入硬件插件(Plugin) --------------------------------------
    InferenceEngine::Core ie;
    std::cout << "\n1. Load plugin..." << std::endl;
    std::cout << ie.GetVersions(DEVICE) << std::endl;
    ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
//    if (DEVICE.find("CPU") != std::string::npos)
//    {
//        ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
//    }


    // ------------------- 2. 读取IR文件 (.xml and .bin files) --------------------------
    std::cout << "\n2. Read IR File..." << std::endl;
    InferenceEngine::CNNNetReader networkReader;
    networkReader.ReadNetwork(xmlFile);  // 读取神经网络模型文件 *.xml
    networkReader.ReadWeights(binFile);  // 读取神经网络权重文件 *.bin
    InferenceEngine::CNNNetwork network = networkReader.getNetwork();
    std::string networkName = networkReader.getName();
    std::cout << "    networkName is " << networkName << std::endl;


    // -------------------- 3. 配置网络输入输出 -----------------------------------------
    std::cout << "\n3. Prepare Input and Output..." << std::endl;
    InferenceEngine::InputsDataMap inputsInfo(network.getInputsInfo());
    std::string imageInputName, imInfoInputName;
    InferenceEngine::InputInfo::Ptr inputInfo = nullptr;
    InferenceEngine::SizeVector inputImageDims;

    /** 遍历模型所有的输入blobs **/
    for (auto & item : inputsInfo)
    {
        /** 处理保存图像数据的第一个张量 **/
        if (item.second->getInputData()->getTensorDesc().getDims().size() == 4)
        {
            /** 处理保存图像数据的第一个张量 **/
            imageInputName = item.first;
            std::cout << "    imageInputName is " << imageInputName << std::endl;
            inputInfo = item.second;
            Precision inputPrecision = InferenceEngine::Precision::FP32;
            item.second->setPrecision(inputPrecision);
            item.second->setLayout(InferenceEngine::Layout::NCHW);
            std::cout << "    Batch size is " << std::to_string(networkReader.getNetwork().getBatchSize()) << std::endl;
        }
//        else if (item.second->getInputData()->getTensorDesc().getDims().size() == 2)
//        {
//            imInfoInputName = item.first;
//            std::cout << "imInfoInputName: " << imInfoInputName << std::endl;
//            Precision inputPrecision = Precision::FP32;
//            item.second->setPrecision(inputPrecision);
//            if ((item.second->getTensorDesc().getDims()[1] != 3) && (item.second->getTensorDesc().getDims()[1] != 6))
//            {
//                throw std::logic_error("Invalid input info. Should be 3 or 6 values length");
//            }
//        }
    }

    /** 获得神经网络的输出信息 **/
    InferenceEngine::OutputsDataMap outputsInfo(network.getOutputsInfo());
    std::string outputName;
    InferenceEngine::DataPtr outputInfo;
    for (const auto &out : outputsInfo)
    {
//        std::cout << out.second->getCreatorLayer().lock()->type << std::endl;
        if (out.second->getCreatorLayer().lock()->type == "Eltwise")  // 对应了xml文件，最后的输出Eltwise，即网络输出的alpha值
        {
            outputName = out.first;
            std::cout << "    outputName is " << outputName << std::endl;
            outputInfo = out.second;
        }
    }

//    const SizeVector outputDims = outputInfo->getTensorDesc().getDims();  // 输出alpha图片的尺寸 [1, 1, 256, 256]
    outputInfo->setPrecision(InferenceEngine::Precision::FP32);


    // --------------------------- 4. 载入模型到AI推理计算设备---------------------------------------
    std::cout << "\n4. Load model into device..." << std::endl;
    InferenceEngine::ExecutableNetwork executable_network = ie.LoadNetwork(network, DEVICE, {});


    // --------------------------- 5. 创建Infer Request--------------------------------------------
    std::cout << "\n5. Create Infer Request..." << std::endl;
    InferenceEngine::InferRequest::Ptr infer_request = executable_network.CreateInferRequestPtr();


    // --------------------------- 6. 准备输入数据 ------------------------------------------------
    std::cout << "\n6. Prepare the input data..." << std::endl;
    cv::Mat img, img_resize;
    img = cv::imread(imageFile);
    const size_t width = (size_t)img.cols;
    const size_t height = (size_t)img.rows;
    std::cout << "    init img channels, width, height = " << img.channels() << ", " << width << ", " << height << std::endl;
    cv::resize(img, img_resize, cv::Size(256, 256), cv::INTER_CUBIC);  // 网络输入尺寸为 256 * 256

    InferenceEngine::Blob::Ptr input = infer_request->GetBlob(network.getInputsInfo().begin()->first);
    auto buffer = input->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();
    cv::Mat tensor;
    img_resize.convertTo(tensor, CV_32FC3);

    // 此步骤根据偏移地址 定义planes分别指向了buffer的数据地址，随后通过split将tensor中的数据根
    // 据RGB三个维度分割给plane，也就将输入数据移到了buffer的数据地址中。
    cv::Size inputLayerSize = cv::Size(256, 256);
    std::vector<cv::Mat> planes(3);
    for (size_t pId = 0; pId < planes.size(); pId++)
    {
        planes[pId] = cv::Mat(cv::Size(256, 256), CV_32FC1, buffer + pId * inputLayerSize.area());
    }
    cv::split(tensor, planes);


    // --------------------------- 7. 执行推理计算 ------------------------------------------------
    std::cout << "\n7.Start inference..." << std::endl;
    std::clock_t begin, end;

    begin = std::clock();
    infer_request->Infer();
    end = std::clock();
//    std::ostringstream infer_time;  // 计算推理计算所花费的时间
    std::cout << "    Infer Time is " << (double)(end - begin) << "ms" << std::endl;


    // --------------------------- 8. 处理输出 ----------------------------------------------------
    std::cout << "\n8.Process output blobs..." << std::endl;
    float *rst = infer_request->GetBlob(outputName)->buffer().as<float*>();

    cv::Mat alpha_fg(256, 256, CV_32FC1, (void *)rst, cv::Mat::AUTO_STEP);  // 将openvino模型的输出，转为opencv mat
    cv::resize(alpha_fg, alpha_fg, cv::Size(width, height), cv::INTER_CUBIC);  // 还原到原始图像大小

    cv::Mat merge_alpha_fg;
    std::vector<cv::Mat> fg_channels;
    fg_channels.push_back(alpha_fg);
    fg_channels.push_back(alpha_fg);
    fg_channels.push_back(alpha_fg);
    cv::merge(fg_channels, merge_alpha_fg);  // 得到3通道的merge_alpha_fg数据

    cv::Mat img_float;
    img.convertTo(img_float, CV_32FC3);
    cv::Mat fg = merge_alpha_fg.mul(img_float);  // 得到 fg 数据


    cv::Mat ones_ = cv::Mat::ones(height, width, CV_32FC1);
    cv::Mat alpha_bg = ones_ - alpha_fg;

    cv::Mat merge_alpha_bg;
    std::vector<cv::Mat> bg_channels_1;
    bg_channels_1.push_back(alpha_bg);
    bg_channels_1.push_back(alpha_bg);
    bg_channels_1.push_back(alpha_bg);
    cv::merge(bg_channels_1, merge_alpha_bg);

    cv::Mat bg_gray = merge_alpha_bg.mul(img_float);
    cv::cvtColor(bg_gray, bg_gray, cv::COLOR_BGR2GRAY);

    cv::Mat bg = img;
    std::vector<cv::Mat> bg_channels_2;
    bg_channels_2.push_back(bg_gray);
    bg_channels_2.push_back(bg_gray);
    bg_channels_2.push_back(bg_gray);
    cv::merge(bg_channels_2, bg);  // 得到 bg 数据


    cv::Mat out = fg + bg;  // 得到最终输出
    end = std::clock();
    std::cout << "    Total time is " << (double)(end - begin) << "ms" << std::endl;

    cv::Mat save;
    out.convertTo(save, CV_8UC1);
    cv::imwrite("save.png", save);

    return 0;
}
