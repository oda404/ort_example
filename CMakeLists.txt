
project(mnist)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_executable(${CMAKE_PROJECT_NAME}
    main.cpp
)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    /home/oda/Documents/cpp/onnxruntime/orttraining/orttraining/training_api/include/
    /home/oda/Documents/cpp/onnxruntime/include/onnxruntime/core/session/
    /usr/include/opencv4
)

target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE
    /home/oda/Documents/cpp/onnxruntime/build/Linux/RelWithDebInfo/
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    onnxruntime
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_imgcodecs
)

