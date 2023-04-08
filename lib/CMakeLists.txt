# MindVision Camera SDK
add_subdirectory(mvsdk)
# Jetson Nano platform
if(CUDAToolkit_FOUND)
    add_subdirectory(TensorRTX_YOLOv5)
endif()
# Add NN inference support for Rockchip RK3588S (Orange pi 5)
if(DEFINED Support_RK3588)
    message(WARNING SSSSSS)
    add_subdirectory(rknpu)
endif()