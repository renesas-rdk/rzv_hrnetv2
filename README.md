# RZV HRNetV2

A C++ package providing HRNetV2-based pose detection models optimized for Renesas RZ/V processors with DRP-AI acceleration support.

## Overview

This package implements HRNetV2 models for human pose estimation.

The classes in this package are intended to be used through the `rzv_model::BaseModel` interface (from dependency rzv_model), and they override the HRNetV2-specific post-processing logic for decoding keypoints and parsing model heatmap outputs.

## Features

* HRNetV2 human pose-estimation model support
* Implements ``rzv_model::HRNetV2Model`` deriving from ``rzv_model::BaseModel``

## Usage

For preparing the AI model files and converting to the DRP-AI compatible format, please refer to the:
- [MMPose Model Zoo](https://github.com/open-mmlab/mmpose)
- [How to convert MMPose models for V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/how_to_convert/How_to_convert_mmpose_models_V2H.md)


### Basic HRNetV2 model
```cpp
#include "rzv_hrnetv2/hrnetv2_model.hpp"

// Create model instance
auto model = std::make_unique<rzv_model::HRNetV2Model>();

// Load model
model->load("path/to/hrnetv2_model");

// Process image
cv::Mat bgr_image = cv::imread("image.png");
// Convert BGR → RGBA
cv::Mat rgba_image;
cv::cvtColor(bgr_image, rgba_image, cv::COLOR_BGR2RGBA);
// Convert RGBA → YUV422 YUYV
cv::Mat yuv422_image = rzv_model::Utils::rgba_to_yuv422(rgba_image, rzv_model::YUV422Format::YUYV);

// Run inference
auto input = rzv_model::ModelInput{yuv422_image, cv::Rect(0, 0, yuv422_image.cols, yuv422_image.rows)};

auto result = model->run<rzv_model::KeyPointResult>(input);

// Process results
if (result && !result->keypoints.empty()) {
    for (const auto &landmark : result->keypoints) {
        std::cout << "KeyPoint -> " << "x: " << landmark.x << ", y: " << landmark.y
                  << ", confidence: " << landmark.confidence << std::endl;
    }
}

```

## ROS2 Integration
This package optionally provides a CMake integration module to simplify usage in ROS2
packages. When enabled, the exported CMake files allow other ROS2 nodes to link against
this model framework using `find_package(rzv_hrnetv2)` and `ament_target_dependencies`.

It can also be built as a standalone C++ library using
make or CMake when ROS2 dependencies (ament and other ROS packages) are removed.


## Dependencies

- rzv_model (base model interface)
- OpenCV

## License
This package is licensed under Apache 2.0.