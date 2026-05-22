// ********************************************************************************************************************
// Copyright [2026] Renesas Electronics Corporation and/or its licensors. All Rights Reserved.
//
// The contents of this file (the "contents") are proprietary and confidential to Renesas Electronics Corporation
// and/or its licensors ("Renesas") and subject to statutory and contractual protections.
//
// Unless otherwise expressly agreed in writing between Renesas and you: 1) you may not use, copy, modify, distribute,
// display, or perform the contents; 2) you may not use any name or mark of Renesas for advertising or publicity
// purposes or in connection with your use of the contents; 3) RENESAS MAKES NO WARRANTY OR REPRESENTATIONS ABOUT THE
// SUITABILITY OF THE CONTENTS FOR ANY PURPOSE; THE CONTENTS ARE PROVIDED "AS IS" WITHOUT ANY EXPRESS OR IMPLIED
// WARRANTY, INCLUDING THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
// NON-INFRINGEMENT; AND 4) RENESAS SHALL NOT BE LIABLE FOR ANY DIRECT, INDIRECT, SPECIAL, OR CONSEQUENTIAL DAMAGES,
// INCLUDING DAMAGES RESULTING FROM LOSS OF USE, DATA, OR PROJECTS, WHETHER IN AN ACTION OF CONTRACT OR TORT, ARISING
// OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE CONTENTS. Third-party contents included in this file may
// be subject to different terms.
// ********************************************************************************************************************
#include "rzv_hrnetv2/hrnetv2_model.hpp"

#include "rzv_model/utils.hpp"

namespace rzv_model
{

class HRNetV2Model::Impl
{
public:
  int num_joints = 21;
  int output_height = 64;
  int output_width = 64;
};

HRNetV2Model::HRNetV2Model() : BaseModel(), pimpl_(std::make_unique<Impl>())
{
  MODEL_INFO("HRNetV2 model instance created with default settings");
}

HRNetV2Model::~HRNetV2Model() = default;

void HRNetV2Model::extract_model_specific_shapes(const ModelShapeInfo & shape_info)
{
  if (!shape_info.output_shapes.empty()) {
    const auto & output_shape = shape_info.output_shapes[0];

    // For HRNetV2, output shape is typically [batch, num_joints, height, width]
    if (output_shape.size() >= 4) {
      pimpl_->num_joints = output_shape[1];
      pimpl_->output_height = output_shape[2];
      pimpl_->output_width = output_shape[3];
    } else if (output_shape.size() == 3) {
      // If output shape is [num_joints, height, width]
      pimpl_->num_joints = output_shape[0];
      pimpl_->output_height = output_shape[1];
      pimpl_->output_width = output_shape[2];
    } else {
      MODEL_ERROR("Unexpected output shape size: {}", output_shape.size());
    }
  }
}

cv::Mat HRNetV2Model::preprocess(const ModelInput & input) { return BaseModel::preprocess(input); }

cv::Mat HRNetV2Model::fallback_preprocess(const ModelInput & input)
{
  return BaseModel::software_preprocess(input, true);
}

std::unique_ptr<ModelResult> HRNetV2Model::postprocess(const std::vector<cv::Mat> & output_tensors)
{
  // HRNetV2 output explanation:
  // - Output is typically a single tensor with heatmaps for each keypoint
  // - Format: [batch, num_joints, height, width] or [num_joints, height, width]
  // - Each channel represents a heatmap for a specific body joint
  // - Higher values in the heatmap indicate higher confidence for joint location

  auto result = std::make_unique<KeyPointResult>();
  result->keypoints.resize(pimpl_->num_joints);

  // Check if we have any output tensors
  if (output_tensors.empty()) {
    MODEL_ERROR("No output tensors received from model");
    return result;
  }

  // For HRNetV2, we typically have a single output tensor with heatmaps
  const cv::Mat & heatmaps = output_tensors[0];

  // Calculate expected dimensions
  const int heatmap_height = pimpl_->output_height;
  const int heatmap_width = pimpl_->output_width;

  // Validate the shape of the tensor
  if (heatmaps.empty()) {
    MODEL_ERROR("Empty heatmaps tensor");
    return result;
  }

  MODEL_DEBUG("Processing heatmaps tensor with dims={}", heatmaps.dims);

  // Process each joint heatmap
  float lowest_score = 1.0f;
  for (int joint = 0; joint < pimpl_->num_joints; ++joint) {
    // Create keypoint with default invalid state
    KeyPoint & kp = result->keypoints[joint];
    kp.class_id = joint;
    kp.confidence = 0.0f;

    // Extract single joint heatmap - the tensor is already properly reshaped by base_model
    cv::Mat heatmap;

    if (heatmaps.dims == 4 && joint < heatmaps.size[1]) {
      // If 4D tensor [batch, joints, height, width], extract the correct slice
      heatmap = cv::Mat(heatmap_height, heatmap_width, CV_32F);
      // Extract the 2D heatmap from the 4D tensor
      const float * src_ptr = heatmaps.ptr<float>(0, joint);
      float * dst_ptr = heatmap.ptr<float>();
      std::memcpy(dst_ptr, src_ptr, heatmap_height * heatmap_width * sizeof(float));
    } else if (heatmaps.dims == 3 && joint < heatmaps.size[0]) {
      // If 3D tensor [joints, height, width], extract the correct slice
      heatmap = cv::Mat(heatmap_height, heatmap_width, CV_32F, (void *)heatmaps.ptr<float>(joint));
    } else {
      MODEL_ERROR("Unexpected heatmap tensor format: dims={}", heatmaps.dims);
      continue;
    }

    // Find the location with maximum confidence in the heatmap
    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(heatmap, &min_val, &max_val, &min_loc, &max_loc);

    // Add sub-pixel refinement if point isn't at border
    float refined_x = max_loc.x;
    float refined_y = max_loc.y;
    if (
      max_loc.x > 0 && max_loc.x < (heatmap_width - 1) && max_loc.y > 0 &&
      max_loc.y < (heatmap_height - 1)) {
      // Compute sub-pixel location by analyzing neighboring pixel differences
      float diff_x =
        heatmap.at<float>(max_loc.y, max_loc.x + 1) - heatmap.at<float>(max_loc.y, max_loc.x - 1);
      float diff_y =
        heatmap.at<float>(max_loc.y + 1, max_loc.x) - heatmap.at<float>(max_loc.y - 1, max_loc.x);

      refined_x += (diff_x > 0 ? 0.25f : -0.25f);
      refined_y += (diff_y > 0 ? 0.25f : -0.25f);
    }

    // Scale point from heatmap dimensions to input dimensions
    // Note: Scale factor of 4 is used because heatmap is typically 1/4 the size of the input
    float x = refined_x * 4;
    float y = refined_y * 4;

    // Map coordinates back to original image space using the same function as YOLOX
    cv::Point2f mapped_point = map_coordinates_to_original(cv::Point2f(x, y));

    // Set the keypoint properties
    kp.x = mapped_point.x;
    kp.y = mapped_point.y;
    kp.confidence = static_cast<float>(max_val);

    MODEL_DEBUG(
      "Detected keypoint {} at ({}, {}) with confidence {}", joint, kp.x, kp.y,
      kp.confidence);

    lowest_score = std::min(lowest_score, kp.confidence);
  }

  // Set overall score based on confidence values
  result->score = lowest_score;
  MODEL_DEBUG("Keypoints detected with overall score: {}", result->score);

  return result;
}

}  // namespace rzv_model