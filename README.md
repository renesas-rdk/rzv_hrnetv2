# RZV HRNetV2

A C++ package providing HRNetV2-based pose detection models optimized for Renesas RZ/V processors with DRP-AI acceleration support.

## Overview

This package implements HRNetV2 models for human pose estimation.

The classes in this package are intended to be used through the `rzv_model::BaseModel` interface (from dependency rzv_model), and they override the HRNetV2-specific post-processing logic for decoding keypoints and parsing model heatmap outputs.

## Features

* HRNetV2 human pose-estimation model support
* Implements ``rzv_model::RTMPoseModel`` deriving from ``rzv_model::BaseModel``

## Usage

For preparing the AI model files and converting to the DRP-AI compatible format, please refer to the ``TBD`` documentation.

### Basic HRNetV2 model
```cpp
#include "rzv_hrnetv2/hrnetv2_model.hpp"

// Create model instance
auto model = std::make_unique<rzv_model::HRNetV2Model>();

// Load model
model->load("path/to/hrnetv2_model");

// Run inference
cv::Mat input_image = cv::imread("image.jpg");
cv::Rect bbox = hand_bbox; //(example: detected hand BBox)

auto model_input = rzv_model::ModelInput{input_image, bbox};

auto model_result = model->run<rzv_model::KeyPointResult>(model_input);

// Process results
if (result && !result->keypoints.empty()) {
    for (const auto &landmark : result->keypoints) {
        std::cout << "KeyPoint -> " << "x: " << landmark.x << ", y: " << landmark.y
                  << ", confidence: " << landmark.confidence << std::endl;
    }
}

```

## Dependencies

- rclcpp
- rzv_model (base model interface)
- OpenCV

## License
This package is licensed under Apache 2.0.