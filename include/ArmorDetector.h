//
// Created by liuzikai on 2/6/21.
//

#ifndef META_VISION_SOLAIS_ARMORDETECTOR_H
#define META_VISION_SOLAIS_ARMORDETECTOR_H

#include "Parameters.h"
#include <mutex>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#ifdef ON_JETSON
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "YOLOv5_TensorRT.h"
#endif
#ifdef Enable_RKNN
#ifndef _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
#define _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM     80
#define NMS_THRESH        0.45
#define BOX_THRESH        0.25
#define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM)

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct __detect_result_t
{
    char name[OBJ_NAME_MAX_SIZE];
    BOX_RECT box;
    float prop;
} detect_result_t;

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);

void deinitPostProcess();
#endif //_RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_

#endif
namespace meta {

class ArmorDetector {
public:

#ifdef ON_JETSON
    ArmorDetector() : yoloModel(std::string("/home/nvidia/tmp/tmp.pS4QeSxQaM/nn-models/model-opt-4.onnx")) {
    }
#endif
    void setParams(const ParamSet &p) { params = p; }

    const ParamSet &getParams() const { return params; }

    struct DetectedArmor {
        std::array<cv::Point2f, 4> points;
        cv::Point2f center;
        bool largeArmor = false;
        int number = 0;                 // not implemented yet
        std::array<int, 2> lightIndex;  // left, right ; already deprecated after YOLO model
        float lightAngleDiff;           // absolute value, non-negative
        float avgLightAngle;
    };

    std::vector<DetectedArmor> detect(const cv::Mat &img);
    std::vector<DetectedArmor> detect_NG(const cv::Mat &img);
    int detect_rknn(int argc, char** argv);

    static float normalizeLightAngle(float angle) { return angle <= 90 ? angle : 180 - angle; }

private:

    ParamSet params;

    cv::Mat imgOriginal;
    cv::Mat imgGray;
    cv::Mat imgBrightness;
    cv::Mat imgColor;
    std::vector<cv::RotatedRect> lightRects;
    cv::Mat imgLights;
#ifdef ON_JETSON
    YOLODet yoloModel;
#endif
    static void drawRotatedRect(cv::Mat &img, const cv::RotatedRect &rect, const cv::Scalar &boarderColor);

    /**
     * Canonicalize a non-square rotated rect from cv::minAreaRect and make:
     *  width: the short edge
     *  height: the long edge
     *  angle: in [0, 180). The angle is then the angle between the long edge and the vertical axis.
     *  https://stackoverflow.com/questions/22696539/reorder-four-points-of-a-rectangle-to-the-correct-order
     * @param rect
     */
    static void canonicalizeRotatedRect(cv::RotatedRect &rect);

    std::vector<DetectedArmor>::iterator filterAcceptedArmorsToRemove(std::vector<DetectedArmor> &acceptedArmors) const;

    friend class Executor;

};

}

#endif //META_VISION_SOLAIS_ARMORDETECTOR_H
