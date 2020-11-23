win32:CONFIG(release, debug|release){LIBS += -LC:/IntelSWTools/openvino_2019.3.334/deployment_tools/inference_engine/lib/intel64/release -linference_engine \
                                                                                                                                          -ltbb \
                                                                                                                                          -ltbbmalloc \
                                                                                                                                          -lcpu_extension \
                                                                                                                                          -lformat_reader \
                                                                                                                                          -lgflags_nothreads_static
#                                     LIBS += -LC:/IntelSWTools/openvino_2019.3.334/deployment_tools/inference_engine/samples/intel64/Release -lcpu_extension -lformat_reader -lgflags_nothreads_static
}
else:win32:CONFIG(debug, debug|release){LIBS += -LC:/IntelSWTools/openvino_2019.3.334/deployment_tools/inference_engine/lib/intel64/debug -linference_engined \
                                                                                                                                           -ltbb_debug \
                                                                                                                                           -ltbbmalloc_debug
                                        LIBS += -LC:/IntelSWTools/openvino_2019.3.334/deployment_tools/inference_engine/samples/intel64/Release -lcpu_extension -lformat_reader -lgflags_nothreads_static
}
#else:unix: LIBS += -LC:/IntelSWTools/openvino_2019.3.334/deployment_tools/inference_engine/lib/intel64/ -linference_engine

INCLUDEPATH += C:/IntelSWTools/openvino_2019.3.334/deployment_tools/inference_engine/include \
               C:/IntelSWTools/openvino_2019.3.334/deployment_tools/inference_engine/samples/common \
               C:/IntelSWTools/openvino_2019.3.334/deployment_tools/inference_engine/src/extension

DEPENDPATH += C:/IntelSWTools/openvino_2019.3.334/deployment_tools/inference_engine/include \
              C:/IntelSWTools/openvino_2019.3.334/deployment_tools/inference_engine/samples/common \
              C:/IntelSWTools/openvino_2019.3.334/deployment_tools/inference_engine/src/extension

win32:CONFIG(release, debug|release){LIBS += -LC:/IntelSWTools/openvino_2019.3.334/opencv/lib/ -lopencv_core412 \
                                                                                        -lopencv_dnn412 \
                                                                                        -lopencv_ml412 \
                                                                                        -lopencv_highgui412 \
                                                                                        -lopencv_imgcodecs412 \
                                                                                        -lopencv_imgproc412 \
                                                                                        -lopencv_video412 \
                                                                                        -lopencv_videoio412
}
else:win32:CONFIG(debug, debug|release){LIBS += -LC:/IntelSWTools/openvino_2019.3.334/opencv/lib/ -lopencv_core412d \
                                                                                        -lopencv_dnn412d \
                                                                                        -lopencv_ml412d \
                                                                                        -lopencv_highgui412d \
                                                                                        -lopencv_imgcodecs412d \
                                                                                        -lopencv_imgproc412d \
                                                                                        -lopencv_video412d \
                                                                                        -lopencv_videoio412d
}

INCLUDEPATH += C:/IntelSWTools/openvino_2019.3.334/opencv/include
DEPENDPATH += C:/IntelSWTools/openvino_2019.3.334/opencv/include
