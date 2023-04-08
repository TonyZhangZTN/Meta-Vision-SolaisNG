//
// Created by liuzikai on 2/6/21.
//

#include "ArmorDetector.h"
#ifdef Enable_RKNN
// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <set>
#include <vector>
#define LABEL_NALE_TXT_PATH "./model/coco_80_labels_list.txt"

static char* labels[OBJ_CLASS_NUM];

const int anchor0[6] = {10, 13, 16, 30, 33, 23};
const int anchor1[6] = {30, 61, 62, 45, 59, 119};
const int anchor2[6] = {116, 90, 156, 198, 373, 326};

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

char* readLine(FILE* fp, char* buffer, int* len)
{
  int    ch;
  int    i        = 0;
  size_t buff_len = 0;

  buffer = (char*)malloc(buff_len + 1);
  if (!buffer)
    return NULL; // Out of memory

  while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
    buff_len++;
    void* tmp = realloc(buffer, buff_len + 1);
    if (tmp == NULL) {
      free(buffer);
      return NULL; // Out of memory
    }
    buffer = (char*)tmp;

    buffer[i] = (char)ch;
    i++;
  }
  buffer[i] = '\0';

  *len = buff_len;

  // Detect end
  if (ch == EOF && (i == 0 || ferror(fp))) {
    free(buffer);
    return NULL;
  }
  return buffer;
}

int readLines(const char* fileName, char* lines[], int max_line)
{
  FILE* file = fopen(fileName, "r");
  char* s;
  int   i = 0;
  int   n = 0;

  if (file == NULL) {
    printf("Open %s fail!\n", fileName);
    return -1;
  }

  while ((s = readLine(file, s, &n)) != NULL) {
    lines[i++] = s;
    if (i >= max_line)
      break;
  }
  fclose(file);
  return i;
}

int loadLabelName(const char* locationFilename, char* label[])
{
  printf("loadLabelName %s\n", locationFilename);
  readLines(locationFilename, label, OBJ_CLASS_NUM);
  return 0;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
  float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
  float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
  float i = w * h;
  float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
  return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds, std::vector<int>& order,
               int filterId, float threshold)
{
  for (int i = 0; i < validCount; ++i) {
    if (order[i] == -1 || classIds[i] != filterId) {
      continue;
    }
    int n = order[i];
    for (int j = i + 1; j < validCount; ++j) {
      int m = order[j];
      if (m == -1 || classIds[i] != filterId) {
        continue;
      }
      float xmin0 = outputLocations[n * 4 + 0];
      float ymin0 = outputLocations[n * 4 + 1];
      float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
      float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

      float xmin1 = outputLocations[m * 4 + 0];
      float ymin1 = outputLocations[m * 4 + 1];
      float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
      float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

      float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

      if (iou > threshold) {
        order[j] = -1;
      }
    }
  }
  return 0;
}

static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices)
{
  float key;
  int   key_index;
  int   low  = left;
  int   high = right;
  if (left < right) {
    key_index = indices[left];
    key       = input[left];
    while (low < high) {
      while (low < high && input[high] <= key) {
        high--;
      }
      input[low]   = input[high];
      indices[low] = indices[high];
      while (low < high && input[low] >= key) {
        low++;
      }
      input[high]   = input[low];
      indices[high] = indices[low];
    }
    input[low]   = key;
    indices[low] = key_index;
    quick_sort_indice_inverse(input, left, low - 1, indices);
    quick_sort_indice_inverse(input, low + 1, right, indices);
  }
  return low;
}

static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline static int32_t __clip(float val, float min, float max)
{
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
  float  dst_val = (f32 / scale) + zp;
  int8_t res     = (int8_t)__clip(dst_val, -128, 127);
  return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static int process(int8_t* input, int* anchor, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float>& boxes, std::vector<float>& objProbs, std::vector<int>& classId, float threshold,
                   int32_t zp, float scale)
{
  int    validCount = 0;
  int    grid_len   = grid_h * grid_w;
  float  thres      = unsigmoid(threshold);
  int8_t thres_i8   = qnt_f32_to_affine(thres, zp, scale);
  for (int a = 0; a < 3; a++) {
    for (int i = 0; i < grid_h; i++) {
      for (int j = 0; j < grid_w; j++) {
        int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
        if (box_confidence >= thres_i8) {
          int     offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
          int8_t* in_ptr = input + offset;
          float   box_x  = sigmoid(deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
          float   box_y  = sigmoid(deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
          float   box_w  = sigmoid(deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
          float   box_h  = sigmoid(deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
          box_x          = (box_x + j) * (float)stride;
          box_y          = (box_y + i) * (float)stride;
          box_w          = box_w * box_w * (float)anchor[a * 2];
          box_h          = box_h * box_h * (float)anchor[a * 2 + 1];
          box_x -= (box_w / 2.0);
          box_y -= (box_h / 2.0);

          int8_t maxClassProbs = in_ptr[5 * grid_len];
          int    maxClassId    = 0;
          for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
            int8_t prob = in_ptr[(5 + k) * grid_len];
            if (prob > maxClassProbs) {
              maxClassId    = k;
              maxClassProbs = prob;
            }
          }
          if (maxClassProbs>thres_i8){
            objProbs.push_back(sigmoid(deqnt_affine_to_f32(maxClassProbs, zp, scale))* sigmoid(deqnt_affine_to_f32(box_confidence, zp, scale)));
            classId.push_back(maxClassId);
            validCount++;
            boxes.push_back(box_x);
            boxes.push_back(box_y);
            boxes.push_back(box_w);
            boxes.push_back(box_h);
          }
        }
      }
    }
  }
  return validCount;
}

int post_process(int8_t* input0, int8_t* input1, int8_t* input2, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, float scale_w, float scale_h, std::vector<int32_t>& qnt_zps,
                 std::vector<float>& qnt_scales, detect_result_group_t* group)
{
  static int init = -1;
  if (init == -1) {
    int ret = 0;
    ret     = loadLabelName(LABEL_NALE_TXT_PATH, labels);
    if (ret < 0) {
      return -1;
    }

    init = 0;
  }
  memset(group, 0, sizeof(detect_result_group_t));

  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int>   classId;

  // stride 8
  int stride0     = 8;
  int grid_h0     = model_in_h / stride0;
  int grid_w0     = model_in_w / stride0;
  int validCount0 = 0;
  validCount0 = process(input0, (int*)anchor0, grid_h0, grid_w0, model_in_h, model_in_w, stride0, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[0], qnt_scales[0]);

  // stride 16
  int stride1     = 16;
  int grid_h1     = model_in_h / stride1;
  int grid_w1     = model_in_w / stride1;
  int validCount1 = 0;
  validCount1 = process(input1, (int*)anchor1, grid_h1, grid_w1, model_in_h, model_in_w, stride1, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[1], qnt_scales[1]);

  // stride 32
  int stride2     = 32;
  int grid_h2     = model_in_h / stride2;
  int grid_w2     = model_in_w / stride2;
  int validCount2 = 0;
  validCount2 = process(input2, (int*)anchor2, grid_h2, grid_w2, model_in_h, model_in_w, stride2, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[2], qnt_scales[2]);

  int validCount = validCount0 + validCount1 + validCount2;
  // no object detect
  if (validCount <= 0) {
    return 0;
  }

  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i) {
    indexArray.push_back(i);
  }

  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c : class_set) {
    nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  int last_count = 0;
  group->count   = 0;
  /* box valid detect target */
  for (int i = 0; i < validCount; ++i) {
    if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
      continue;
    }
    int n = indexArray[i];

    float x1       = filterBoxes[n * 4 + 0];
    float y1       = filterBoxes[n * 4 + 1];
    float x2       = x1 + filterBoxes[n * 4 + 2];
    float y2       = y1 + filterBoxes[n * 4 + 3];
    int   id       = classId[n];
    float obj_conf = objProbs[i];

    group->results[last_count].box.left   = (int)(clamp(x1, 0, model_in_w) / scale_w);
    group->results[last_count].box.top    = (int)(clamp(y1, 0, model_in_h) / scale_h);
    group->results[last_count].box.right  = (int)(clamp(x2, 0, model_in_w) / scale_w);
    group->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
    group->results[last_count].prop       = obj_conf;
    char* label                           = labels[id];
    strncpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE);

    // printf("result %2d: (%4d, %4d, %4d, %4d), %s\n", i, group->results[last_count].box.left,
    // group->results[last_count].box.top,
    //        group->results[last_count].box.right, group->results[last_count].box.bottom, label);
    last_count++;
  }
  group->count = last_count;

  return 0;
}

void deinitPostProcess()
{
  for (int i = 0; i < OBJ_CLASS_NUM; i++) {
    if (labels[i] != nullptr) {
      free(labels[i]);
      labels[i] = nullptr;
    }
  }
}
// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define _BASETSD_H

#include <RgaUtils.h>
#include <im2d.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rga.h>
#include <rknn_api.h>

#define PERF_WITH_POST 1
/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz)
{
  unsigned char* data;
  int            ret;

  data = NULL;

  if (NULL == fp) {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0) {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char*)malloc(sz);
  if (data == NULL) {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char* load_model(const char* filename, int* model_size)
{
  FILE*          fp;
  unsigned char* data;

  fp = fopen(filename, "rb");
  if (NULL == fp) {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

static int saveFloat(const char* file_name, float* output, int element_size)
{
  FILE* fp;
  fp = fopen(file_name, "w");
  for (int i = 0; i < element_size; i++) {
    fprintf(fp, "%.6f\n", output[i]);
  }
  fclose(fp);
  return 0;
}

#endif


namespace meta {
#ifdef Enable_RKNN

int ArmorDetector::detect_rknn(int argc, char** argv)
    {
        int            status     = 0;
        char*          model_name = NULL;
        rknn_context   ctx;
        size_t         actual_size        = 0;
        int            img_width          = 0;
        int            img_height         = 0;
        int            img_channel        = 0;
        const float    nms_threshold      = NMS_THRESH;
        const float    box_conf_threshold = BOX_THRESH;
        struct timeval start_time, stop_time;
        int            ret;

        // init rga context
        rga_buffer_t src;
        rga_buffer_t dst;
        im_rect      src_rect;
        im_rect      dst_rect;
        memset(&src_rect, 0, sizeof(src_rect));
        memset(&dst_rect, 0, sizeof(dst_rect));
        memset(&src, 0, sizeof(src));
        memset(&dst, 0, sizeof(dst));

        if (argc != 3) {
            printf("Usage: %s <rknn model> <jpg> \n", argv[0]);
            return -1;
        }

        printf("post process config: box_conf_threshold = %.2f, nms_threshold = %.2f\n", box_conf_threshold, nms_threshold);

        model_name       = (char*)argv[1];
        char* image_name = argv[2];

        printf("Read %s ...\n", image_name);
        cv::Mat orig_img = cv::imread(image_name, 1);
        if (!orig_img.data) {
            printf("cv::imread %s fail!\n", image_name);
            return -1;
        }
        cv::Mat img;
        cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
        img_width  = img.cols;
        img_height = img.rows;
        printf("img width = %d, img height = %d\n", img_width, img_height);

        /* Create the neural network */
        printf("Loading mode...\n");
        int            model_data_size = 0;
        unsigned char* model_data      = load_model(model_name, &model_data_size);
        ret                            = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
        if (ret < 0) {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }

        rknn_sdk_version version;
        ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
        if (ret < 0) {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

        rknn_input_output_num io_num;
        ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
        if (ret < 0) {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

        rknn_tensor_attr input_attrs[io_num.n_input];
        memset(input_attrs, 0, sizeof(input_attrs));
        for (int i = 0; i < io_num.n_input; i++) {
            input_attrs[i].index = i;
            ret                  = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
            if (ret < 0) {
                printf("rknn_init error ret=%d\n", ret);
                return -1;
            }
            dump_tensor_attr(&(input_attrs[i]));
        }

        rknn_tensor_attr output_attrs[io_num.n_output];
        memset(output_attrs, 0, sizeof(output_attrs));
        for (int i = 0; i < io_num.n_output; i++) {
            output_attrs[i].index = i;
            ret                   = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
            dump_tensor_attr(&(output_attrs[i]));
        }

        int channel = 3;
        int width   = 0;
        int height  = 0;
        if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
            printf("model is NCHW input fmt\n");
            channel = input_attrs[0].dims[1];
            height  = input_attrs[0].dims[2];
            width   = input_attrs[0].dims[3];
        } else {
            printf("model is NHWC input fmt\n");
            height  = input_attrs[0].dims[1];
            width   = input_attrs[0].dims[2];
            channel = input_attrs[0].dims[3];
        }

        printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index        = 0;
        inputs[0].type         = RKNN_TENSOR_UINT8;
        inputs[0].size         = width * height * channel;
        inputs[0].fmt          = RKNN_TENSOR_NHWC;
        inputs[0].pass_through = 0;

        // You may not need resize when src resulotion equals to dst resulotion
        void* resize_buf = nullptr;

        if (img_width != width || img_height != height) {
            printf("resize with RGA!\n");
            resize_buf = malloc(height * width * channel);
            memset(resize_buf, 0x00, height * width * channel);

            src = wrapbuffer_virtualaddr((void*)img.data, img_width, img_height, RK_FORMAT_RGB_888);
            dst = wrapbuffer_virtualaddr((void*)resize_buf, width, height, RK_FORMAT_RGB_888);
            ret = imcheck(src, dst, src_rect, dst_rect);
            if (IM_STATUS_NOERROR != ret) {
                printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
                return -1;
            }
            IM_STATUS STATUS = imresize(src, dst);

            // for debug
            cv::Mat resize_img(cv::Size(width, height), CV_8UC3, resize_buf);
            cv::imwrite("resize_input.jpg", resize_img);

            inputs[0].buf = resize_buf;
        } else {
            inputs[0].buf = (void*)img.data;
        }

        gettimeofday(&start_time, NULL);
        rknn_inputs_set(ctx, io_num.n_input, inputs);

        rknn_output outputs[io_num.n_output];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < io_num.n_output; i++) {
            outputs[i].want_float = 0;
        }

        ret = rknn_run(ctx, NULL);
        ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
        gettimeofday(&stop_time, NULL);
        printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

        // post process
        float scale_w = (float)width / img_width;
        float scale_h = (float)height / img_height;

        detect_result_group_t detect_result_group;
        std::vector<float>    out_scales;
        std::vector<int32_t>  out_zps;
        for (int i = 0; i < io_num.n_output; ++i) {
            out_scales.push_back(output_attrs[i].scale);
            out_zps.push_back(output_attrs[i].zp);
        }
        post_process((int8_t*)outputs[0].buf, (int8_t*)outputs[1].buf, (int8_t*)outputs[2].buf, height, width,
                     box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

        // Draw Objects
        char text[256];
        for (int i = 0; i < detect_result_group.count; i++) {
            detect_result_t* det_result = &(detect_result_group.results[i]);
            sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
            printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
                   det_result->box.right, det_result->box.bottom, det_result->prop);
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;
            rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);
            putText(orig_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

        imwrite("./out.jpg", orig_img);
        ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

        // loop test
        int test_count = 10;
        gettimeofday(&start_time, NULL);
        for (int i = 0; i < test_count; ++i) {
            rknn_inputs_set(ctx, io_num.n_input, inputs);
            ret = rknn_run(ctx, NULL);
            ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
#if PERF_WITH_POST
            post_process((int8_t*)outputs[0].buf, (int8_t*)outputs[1].buf, (int8_t*)outputs[2].buf, height, width,
                         box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
#endif
            ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
        }
        gettimeofday(&stop_time, NULL);
        printf("loop count = %d , average run  %f ms\n", test_count,
               (__get_us(stop_time) - __get_us(start_time)) / 1000.0 / test_count);

        deinitPostProcess();

        // release
        ret = rknn_destroy(ctx);

        if (model_data) {
            free(model_data);
        }

        if (resize_buf) {
            free(resize_buf);
        }

        return 0;
    }
#endif
std::vector<ArmorDetector::DetectedArmor> ArmorDetector::detect(const cv::Mat &img) {

    /*
     * Note: in this mega function, steps are wrapped with {} to reduce local variable pollution and make it easier to
     *       read. Please follow the convention.
     * Note: for debug purpose, allows some local variables and single-line compound statements.
     */

    // ================================ Setup ================================
    {
        imgOriginal = img;
        imgGray = imgBrightness = imgColor = imgLights = cv::Mat();
    }

    // ================================ Brightness Threshold ================================
    {
        cvtColor(imgOriginal, imgGray, cv::COLOR_BGR2GRAY);
        threshold(imgGray, imgBrightness, params.brightness_threshold(), 255, cv::THRESH_BINARY);
    }

    // ================================ Color Threshold ================================
    {
        if (params.color_threshold_mode() == ParamSet::HSV) {

            // Convert to HSV color space
            cv::Mat hsvImg;
            cvtColor(imgOriginal, hsvImg, cv::COLOR_BGR2HSV);

            if (params.enemy_color() == ParamSet::RED) {
                // Red color spreads over the 0 (180) boundary, so combine them
                cv::Mat thresholdImg0, thresholdImg1;
                inRange(hsvImg, cv::Scalar(0, 0, 0), cv::Scalar(params.hsv_red_hue().max(), 255, 255), thresholdImg0);
                inRange(hsvImg, cv::Scalar(params.hsv_red_hue().min(), 0, 0), cv::Scalar(180, 255, 255), thresholdImg1);
                imgColor = thresholdImg0 | thresholdImg1;
            } else {
                inRange(hsvImg, cv::Scalar(params.hsv_blue_hue().min(), 0, 0),
                        cv::Scalar(params.hsv_blue_hue().max(), 255, 255),
                        imgColor);
            }

        } else {

            std::vector<cv::Mat> channels;
            split(imgOriginal, channels);

            // Filter using channel subtraction
            int mainChannel = (params.enemy_color() == ParamSet::RED ? 2 : 0);
            int oppositeChannel = (params.enemy_color() == ParamSet::RED ? 0 : 2);
            subtract(channels[mainChannel], channels[oppositeChannel], imgColor);
            threshold(imgColor, imgColor, params.rb_channel_threshold(), 255, cv::THRESH_BINARY);

        }

        // Color erode
        if (params.contour_erode().enabled()) {
            cv::Mat element = cv::getStructuringElement(
                    cv::MORPH_ELLIPSE,
                    cv::Size(params.contour_erode().val(), params.contour_erode().val()));
            erode(imgColor, imgColor, element);
        }

        // Color dilate
        if (params.contour_dilate().enabled()) {
            cv::Mat element = cv::getStructuringElement(
                    cv::MORPH_ELLIPSE,
                    cv::Size(params.contour_dilate().val(), params.contour_dilate().val()));
            dilate(imgColor, imgColor, element);
        }

        // Apply filter
        imgLights = imgBrightness & imgColor;
    }

    // ================================ Find Contours ================================

    // Contour open
    if (params.contour_open().enabled()) {
        cv::Mat element = cv::getStructuringElement(
                cv::MORPH_ELLIPSE,
                cv::Size(params.contour_open().val(), params.contour_open().val()));
        morphologyEx(imgLights, imgLights, cv::MORPH_OPEN, element);
    }

    // Contour close
    if (params.contour_close().enabled()) {
        cv::Mat element = cv::getStructuringElement(
                cv::MORPH_ELLIPSE,
                cv::Size(params.contour_close().val(), params.contour_close().val()));
        morphologyEx(imgLights, imgLights, cv::MORPH_CLOSE, element);
    }

    {
        lightRects.clear();

        std::vector<std::vector<cv::Point>> contours;
        findContours(imgLights, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Filter individual contours
        for (const auto &contour : contours) {

            // Filter pixel count
            if (params.contour_pixel_count().enabled()) {
                if (contour.size() < params.contour_pixel_count().val()) {
                    continue;
                }
            }

            // Filter area size
            if (params.contour_min_area().enabled()) {
                double area = contourArea(contour);
                if (area < params.contour_min_area().val()) {
                    continue;
                }
            }

            // Fit contour using a rotated rect
            cv::RotatedRect rect;
            switch (params.contour_fit_function()) {
                case ParamSet::MIN_AREA_RECT:
                    rect = minAreaRect(contour);
                    break;
                case ParamSet::ELLIPSE:
                    // There should be at least 5 points to fit the ellipse
                    if (contour.size() < 5) continue;
                    rect = fitEllipse(contour);
                    break;
                case ParamSet::ELLIPSE_AMS:
                    if (contour.size() < 5) continue;
                    rect = fitEllipseAMS(contour);
                    break;
                case ParamSet::ELLIPSE_DIRECT:
                    if (contour.size() < 5) continue;
                    rect = fitEllipseDirect(contour);
                    break;
                default:
                    assert(!"Invalid params.contour_fit_function()");
            }
            canonicalizeRotatedRect(rect);
            // Now, width: the short edge, height: the long edge, angle: in [0, 180)

            // Filter long edge min length
            if (params.long_edge_min_length().enabled() && rect.size.height < params.long_edge_min_length().val()) {
                continue;
            }

            // Filter angle
            if (params.light_max_rotation().enabled() &&
                std::min(rect.angle, 180 - rect.angle) >= params.light_max_rotation().val()) {
                continue;
            }

            // Filter aspect ratio
            if (params.light_aspect_ratio().enabled()) {
                double aspectRatio = rect.size.height / rect.size.width;
                if (!inRange(aspectRatio, params.light_aspect_ratio())) {
                    continue;
                }
            }

            // Accept the rect
            lightRects.emplace_back(rect);
        }
    }

    // If there is less than two light contours, stop detection
    if (lightRects.size() < 2) {
        return {};
    }

    // Sort lights from left to right based on center X
    sort(lightRects.begin(), lightRects.end(),
         [](cv::RotatedRect &a1, cv::RotatedRect &a2) {
             return a1.center.x < a2.center.x;
         });

    /*
     * OpenCV coordinate: +x right, +y down
     */

    // ================================ Combine Lights to Armors ================================
    std::vector<DetectedArmor> acceptedArmors;
    {
        std::array<cv::Point2f, 4> armorPoints;
        /*
         *              1 ----------- 2
         *            |*|             |*|
         * left light |*|             |*| right light
         *            |*|             |*|
         *              0 ----------- 3
         *
         * Edges (0, 1) and (2, 3) lie on inner edge
         */

        for (int leftLightIndex = 0; leftLightIndex < lightRects.size() - 1; ++leftLightIndex) {

            const cv::RotatedRect &leftRect = lightRects[leftLightIndex];  // already canonicalized


            cv::Point2f leftPoints[4];
            leftRect.points(leftPoints);  // bottomLeft, topLeft, topRight, bottomRight of unrotated rect
            if (leftRect.angle <= 90) {
                armorPoints[0] = (leftPoints[0] + leftPoints[3]) / 2;
                armorPoints[1] = (leftPoints[1] + leftPoints[2]) / 2;
            } else {
                armorPoints[0] = (leftPoints[1] + leftPoints[2]) / 2;
                armorPoints[1] = (leftPoints[0] + leftPoints[3]) / 2;
            }

            auto &leftCenter = leftRect.center;

            for (int rightLightIndex = leftLightIndex + 1; rightLightIndex < lightRects.size(); rightLightIndex++) {

                const cv::RotatedRect &rightRect = lightRects[rightLightIndex];  // already canonicalized


                cv::Point2f rightPoints[4];
                rightRect.points(rightPoints);  // bottomLeft, topLeft, topRight, bottomRight of unrotated rect
                if (rightRect.angle <= 90) {
                    armorPoints[3] = (rightPoints[0] + rightPoints[3]) / 2;
                    armorPoints[2] = (rightPoints[1] + rightPoints[2]) / 2;
                } else {
                    armorPoints[3] = (rightPoints[1] + rightPoints[2]) / 2;
                    armorPoints[2] = (rightPoints[0] + rightPoints[3]) / 2;
                }


                auto leftVector = armorPoints[1] - armorPoints[0];   // up
                if (leftVector.y > 0) {
                    continue;  // leftVector should be upward, or lights intersect
                }
                auto rightVector = armorPoints[2] - armorPoints[3];  // up
                if (rightVector.y > 0) {
                    continue;  // rightVector should be upward, or lights intersect
                }
                auto topVector = armorPoints[2] - armorPoints[1];    // right
                if (topVector.x < 0) {
                    continue;  // topVector should be rightward, or lights intersect
                }
                auto bottomVector = armorPoints[3] - armorPoints[0];  // right
                if (bottomVector.x < 0) {
                    continue;  // bottomVector should be rightward, or lights intersect
                }


                auto &rightCenter = rightRect.center;

                double leftLength = cv::norm(armorPoints[1] - armorPoints[0]);
                double rightLength = cv::norm(armorPoints[2] - armorPoints[3]);
                double averageLength = (leftLength + rightLength) / 2;

                // Filter long light length to short light length ratio
                if (params.light_length_max_ratio().enabled()) {
                    double lengthRatio = leftLength > rightLength ?
                                         leftLength / rightLength : rightLength / leftLength;  // >= 1
                    if (lengthRatio > params.light_length_max_ratio().val()) continue;
                }

                // Filter central X's difference
                if (params.light_x_dist_over_l().enabled()) {
                    double xDiffOverAvgL = abs(leftCenter.x - rightCenter.x) / averageLength;
                    if (!inRange(xDiffOverAvgL, params.light_x_dist_over_l())) {
                        continue;
                    }
                }

                // Filter central Y's difference
                if (params.light_y_dist_over_l().enabled()) {
                    double yDiffOverAvgL = abs(leftCenter.y - rightCenter.y) / averageLength;
                    if (!inRange(yDiffOverAvgL, params.light_y_dist_over_l())) {
                        continue;
                    }
                }

                // Filter angle difference
                float angleDiff = std::abs(leftRect.angle - rightRect.angle);
                if (params.light_angle_max_diff().enabled()) {
                    if (angleDiff > 90) {
                        angleDiff = 180 - angleDiff;
                    }
                    if (angleDiff > params.light_angle_max_diff().val()) {
                        continue;
                    }
                }

                double armorHeight = (cv::norm(leftVector) + cv::norm(rightVector)) / 2;
                double armorWidth = (cv::norm(topVector) + cv::norm(bottomVector)) / 2;

                // Filter armor aspect ratio
                bool largeArmor;
                if (inRange(armorWidth / armorHeight, params.small_armor_aspect_ratio())) {
                    largeArmor = false;
                } else if (inRange(armorWidth / armorHeight, params.large_armor_aspect_ratio())) {
                    largeArmor = true;
                } else {
                    continue;
                }


                // Accept the armor
                cv::Point2f center = {0, 0};
                for (int i = 0; i < 4; i++) {
                    center.x += armorPoints[i].x;
                    center.y += armorPoints[i].y;
                }

                // Just use the average X and Y coordinate for the four point
                center.x /= 4;
                center.y /= 4;

                acceptedArmors.emplace_back(DetectedArmor{
                        armorPoints,
                        center,
                        largeArmor,
                        0,
                        {leftLightIndex, rightLightIndex},
                        angleDiff,
                        (normalizeLightAngle(leftRect.angle) + normalizeLightAngle(rightRect.angle)) / 2
                });
            }
        }
    }

    // Filter armors that share lights
    {
        while(true) {
            auto it = filterAcceptedArmorsToRemove(acceptedArmors);
            if (it == acceptedArmors.end()) break;  // nothing to remove
            acceptedArmors.erase(it);       // remove the armor
            // continue to try again
        }
    }

    return acceptedArmors;
}
#ifdef ON_JETSON
std::vector<ArmorDetector::DetectedArmor> ArmorDetector::detect_NG(const cv::Mat &img) {
    // ================================ Setup ================================
    {
        imgOriginal = img;
        imgGray = imgBrightness = imgColor = imgLights = cv::Mat();
    }

    // ================================ Brightness Threshold ================================
    {
        cvtColor(imgOriginal, imgGray, cv::COLOR_BGR2GRAY);
        threshold(imgGray, imgBrightness, params.brightness_threshold(), 255, cv::THRESH_BINARY);
    }

    // ================================ Color Threshold ================================
    {
        if (params.color_threshold_mode() == ParamSet::HSV) {

            // Convert to HSV color space
            cv::Mat hsvImg;
            cvtColor(imgOriginal, hsvImg, cv::COLOR_BGR2HSV);

            if (params.enemy_color() == ParamSet::RED) {
                // Red color spreads over the 0 (180) boundary, so combine them
                cv::Mat thresholdImg0, thresholdImg1;
                inRange(hsvImg, cv::Scalar(0, 0, 0), cv::Scalar(params.hsv_red_hue().max(), 255, 255), thresholdImg0);
                inRange(hsvImg, cv::Scalar(params.hsv_red_hue().min(), 0, 0), cv::Scalar(180, 255, 255), thresholdImg1);
                imgColor = thresholdImg0 | thresholdImg1;
            } else {
                inRange(hsvImg, cv::Scalar(params.hsv_blue_hue().min(), 0, 0),
                        cv::Scalar(params.hsv_blue_hue().max(), 255, 255),
                        imgColor);
            }

        } else {

            std::vector<cv::Mat> channels;
            split(imgOriginal, channels);

            // Filter using channel subtraction
            int mainChannel = (params.enemy_color() == ParamSet::RED ? 2 : 0);
            int oppositeChannel = (params.enemy_color() == ParamSet::RED ? 0 : 2);
            subtract(channels[mainChannel], channels[oppositeChannel], imgColor);
            threshold(imgColor, imgColor, params.rb_channel_threshold(), 255, cv::THRESH_BINARY);

        }

        // Color erode
        if (params.contour_erode().enabled()) {
            cv::Mat element = cv::getStructuringElement(
                    cv::MORPH_ELLIPSE,
                    cv::Size(params.contour_erode().val(), params.contour_erode().val()));
            erode(imgColor, imgColor, element);
        }

        // Color dilate
        if (params.contour_dilate().enabled()) {
            cv::Mat element = cv::getStructuringElement(
                    cv::MORPH_ELLIPSE,
                    cv::Size(params.contour_dilate().val(), params.contour_dilate().val()));
            dilate(imgColor, imgColor, element);
        }

        // Apply filter
        imgLights = imgBrightness & imgColor;
    }

    // ================================ Find Contours ================================

    // Contour open
    if (params.contour_open().enabled()) {
        cv::Mat element = cv::getStructuringElement(
                cv::MORPH_ELLIPSE,
                cv::Size(params.contour_open().val(), params.contour_open().val()));
        morphologyEx(imgLights, imgLights, cv::MORPH_OPEN, element);
    }

    // Contour close
    if (params.contour_close().enabled()) {
        cv::Mat element = cv::getStructuringElement(
                cv::MORPH_ELLIPSE,
                cv::Size(params.contour_close().val(), params.contour_close().val()));
        morphologyEx(imgLights, imgLights, cv::MORPH_CLOSE, element);
    }

    {
        lightRects.clear();

        std::vector<std::vector<cv::Point>> contours;
        findContours(imgLights, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Filter individual contours
        for (const auto &contour : contours) {

            // Filter pixel count
            if (params.contour_pixel_count().enabled()) {
                if (contour.size() < params.contour_pixel_count().val()) {
                    continue;
                }
            }

            // Filter area size
            if (params.contour_min_area().enabled()) {
                double area = contourArea(contour);
                if (area < params.contour_min_area().val()) {
                    continue;
                }
            }

            // Fit contour using a rotated rect
            cv::RotatedRect rect;
            switch (params.contour_fit_function()) {
                case ParamSet::MIN_AREA_RECT:
                    rect = minAreaRect(contour);
                    break;
                case ParamSet::ELLIPSE:
                    // There should be at least 5 points to fit the ellipse
                    if (contour.size() < 5) continue;
                    rect = fitEllipse(contour);
                    break;
                case ParamSet::ELLIPSE_AMS:
                    if (contour.size() < 5) continue;
                    rect = fitEllipseAMS(contour);
                    break;
                case ParamSet::ELLIPSE_DIRECT:
                    if (contour.size() < 5) continue;
                    rect = fitEllipseDirect(contour);
                    break;
                default:
                    assert(!"Invalid params.contour_fit_function()");
            }
            canonicalizeRotatedRect(rect);
            // Now, width: the short edge, height: the long edge, angle: in [0, 180)

            // Filter long edge min length
            if (params.long_edge_min_length().enabled() && rect.size.height < params.long_edge_min_length().val()) {
                continue;
            }

            // Filter angle
            if (params.light_max_rotation().enabled() &&
                std::min(rect.angle, 180 - rect.angle) >= params.light_max_rotation().val()) {
                continue;
            }

            // Filter aspect ratio
            if (params.light_aspect_ratio().enabled()) {
                double aspectRatio = rect.size.height / rect.size.width;
                if (!inRange(aspectRatio, params.light_aspect_ratio())) {
                    continue;
                }
            }

            // Accept the rect
            lightRects.emplace_back(rect);
        }
    }

    // If there is less than two light contours, stop detection
    if (lightRects.size() < 2) {
        return {};
    }

    // Sort lights from left to right based on center X
    sort(lightRects.begin(), lightRects.end(),
         [](cv::RotatedRect &a1, cv::RotatedRect &a2) {
             return a1.center.x < a2.center.x;
         });

    /*
     * OpenCV coordinate: +x right, +y down
     */

    // ================================ Combine Lights to Armors ================================
    std::vector<DetectedArmor> acceptedArmors;
    {
        std::array<cv::Point2f, 4> armorPoints;
        /*
         *              1 ----------- 2
         *            |*|             |*|
         * left light |*|             |*| right light
         *            |*|             |*|
         *              0 ----------- 3
         *
         * Edges (0, 1) and (2, 3) lie on inner edge
         */

        for (int leftLightIndex = 0; leftLightIndex < lightRects.size() - 1; ++leftLightIndex) {

            const cv::RotatedRect &leftRect = lightRects[leftLightIndex];  // already canonicalized


            cv::Point2f leftPoints[4];
            leftRect.points(leftPoints);  // bottomLeft, topLeft, topRight, bottomRight of unrotated rect
            if (leftRect.angle <= 90) {
                armorPoints[0] = (leftPoints[0] + leftPoints[3]) / 2;
                armorPoints[1] = (leftPoints[1] + leftPoints[2]) / 2;
            } else {
                armorPoints[0] = (leftPoints[1] + leftPoints[2]) / 2;
                armorPoints[1] = (leftPoints[0] + leftPoints[3]) / 2;
            }

            auto &leftCenter = leftRect.center;

            for (int rightLightIndex = leftLightIndex + 1; rightLightIndex < lightRects.size(); rightLightIndex++) {

                const cv::RotatedRect &rightRect = lightRects[rightLightIndex];  // already canonicalized


                cv::Point2f rightPoints[4];
                rightRect.points(rightPoints);  // bottomLeft, topLeft, topRight, bottomRight of unrotated rect
                if (rightRect.angle <= 90) {
                    armorPoints[3] = (rightPoints[0] + rightPoints[3]) / 2;
                    armorPoints[2] = (rightPoints[1] + rightPoints[2]) / 2;
                } else {
                    armorPoints[3] = (rightPoints[1] + rightPoints[2]) / 2;
                    armorPoints[2] = (rightPoints[0] + rightPoints[3]) / 2;
                }


                auto leftVector = armorPoints[1] - armorPoints[0];   // up
                if (leftVector.y > 0) {
                    continue;  // leftVector should be upward, or lights intersect
                }
                auto rightVector = armorPoints[2] - armorPoints[3];  // up
                if (rightVector.y > 0) {
                    continue;  // rightVector should be upward, or lights intersect
                }
                auto topVector = armorPoints[2] - armorPoints[1];    // right
                if (topVector.x < 0) {
                    continue;  // topVector should be rightward, or lights intersect
                }
                auto bottomVector = armorPoints[3] - armorPoints[0];  // right
                if (bottomVector.x < 0) {
                    continue;  // bottomVector should be rightward, or lights intersect
                }


                auto &rightCenter = rightRect.center;

                double leftLength = cv::norm(armorPoints[1] - armorPoints[0]);
                double rightLength = cv::norm(armorPoints[2] - armorPoints[3]);
                double averageLength = (leftLength + rightLength) / 2;

                // Filter long light length to short light length ratio
                if (params.light_length_max_ratio().enabled()) {
                    double lengthRatio = leftLength > rightLength ?
                                         leftLength / rightLength : rightLength / leftLength;  // >= 1
                    if (lengthRatio > params.light_length_max_ratio().val()) continue;
                }

                // Filter central X's difference
                if (params.light_x_dist_over_l().enabled()) {
                    double xDiffOverAvgL = abs(leftCenter.x - rightCenter.x) / averageLength;
                    if (!inRange(xDiffOverAvgL, params.light_x_dist_over_l())) {
                        continue;
                    }
                }

                // Filter central Y's difference
                if (params.light_y_dist_over_l().enabled()) {
                    double yDiffOverAvgL = abs(leftCenter.y - rightCenter.y) / averageLength;
                    if (!inRange(yDiffOverAvgL, params.light_y_dist_over_l())) {
                        continue;
                    }
                }

                // Filter angle difference
                float angleDiff = std::abs(leftRect.angle - rightRect.angle);
                if (params.light_angle_max_diff().enabled()) {
                    if (angleDiff > 90) {
                        angleDiff = 180 - angleDiff;
                    }
                    if (angleDiff > params.light_angle_max_diff().val()) {
                        continue;
                    }
                }

                double armorHeight = (cv::norm(leftVector) + cv::norm(rightVector)) / 2;
                double armorWidth = (cv::norm(topVector) + cv::norm(bottomVector)) / 2;

                // Filter armor aspect ratio
                bool largeArmor;
                if (inRange(armorWidth / armorHeight, params.small_armor_aspect_ratio())) {
                    largeArmor = false;
                } else if (inRange(armorWidth / armorHeight, params.large_armor_aspect_ratio())) {
                    largeArmor = true;
                } else {
                    continue;
                }


                // Accept the armor
                cv::Point2f center = {0, 0};
                for (int i = 0; i < 4; i++) {
                    center.x += armorPoints[i].x;
                    center.y += armorPoints[i].y;
                }

                // Just use the average X and Y coordinate for the four point
                center.x /= 4;
                center.y /= 4;

                acceptedArmors.emplace_back(DetectedArmor{
                        armorPoints,
                        center,
                        largeArmor,
                        0,
                        {leftLightIndex, rightLightIndex},
                        angleDiff,
                        (normalizeLightAngle(leftRect.angle) + normalizeLightAngle(rightRect.angle)) / 2
                });
            }
        }
    }

    // Filter armors that share lights
    {
        while(true) {
            auto it = filterAcceptedArmorsToRemove(acceptedArmors);
            if (it == acceptedArmors.end()) break;  // nothing to remove
            acceptedArmors.erase(it);       // remove the armor
            // continue to try again
        }
    }

    std::vector<YOLODet::bbox_t> detectResults = yoloModel(imgOriginal);
    std::vector<DetectedArmor> acceptedArmors_NG;

    int cnt = 0;
    for (auto armor : detectResults) {
        std::cout << "Armor " << cnt << ": " << armor.color_id << " " << armor.pts[0] << " " << armor.pts[1] << " " << armor.pts[2] << " " << armor.pts[3] << std::endl;
        cnt++;
        if (armor.color_id == !(params.enemy_color())) {
            std::array<cv::Point2f, 4> armorPoints = armor.pts;

            auto leftVector = armorPoints[1] - armorPoints[0];      // pointing up
            auto rightVector = armorPoints[2] - armorPoints[3];     // pointing up
            auto topVector = armorPoints[2] - armorPoints[1];       // pointing right
            auto bottomVector = armorPoints[3] - armorPoints[0];    // pointing right

            double armorHeight = (cv::norm(leftVector) + cv::norm(rightVector)) / 2;
            double armorWidth = (cv::norm(topVector) + cv::norm(bottomVector)) / 2;

            bool largeArmor;
            if (inRange(armorWidth / armorHeight, params.small_armor_aspect_ratio())) {
                largeArmor = false;
            } else if (inRange(armorWidth / armorHeight, params.large_armor_aspect_ratio())) {
                largeArmor = true;
            } else {
                continue;
            }

            cv::Point2f center = {0, 0};
            for (int i = 0; i < 4; i++) {
                center.x += armorPoints[i].x;
                center.y += armorPoints[i].y;
            }
            // Just use the average X and Y coordinate for the four point
            center.x /= 4;
            center.y /= 4;

            float angleDiff = std::abs(cv::fastAtan2(leftVector.y, leftVector.x) - cv::fastAtan2(rightVector.y, rightVector.x));

            acceptedArmors_NG.push_back(DetectedArmor{armorPoints,
                                                      center,
                                                      largeArmor,
                                                      0,
                                                      {0 ,0},
                                                      angleDiff,
                                                      (normalizeLightAngle(cv::fastAtan2(leftVector.y, leftVector.x)) +\
                                                       normalizeLightAngle(cv::fastAtan2(rightVector.y, rightVector.x))) / 2});
        }
    }



//    return acceptedArmors;
    return acceptedArmors_NG;
}
#endif
std::vector<ArmorDetector::DetectedArmor>::iterator
ArmorDetector::filterAcceptedArmorsToRemove(std::vector<DetectedArmor> &acceptedArmors) const {
    for (auto it = acceptedArmors.begin(); it != acceptedArmors.end(); ++it) {
        for (auto it2 = it + 1; it2 != acceptedArmors.end(); ++it2) {
            if (it->lightIndex[0] == it2->lightIndex[0] || it->lightIndex[0] == it2->lightIndex[1] ||
                it->lightIndex[1] == it2->lightIndex[0] || it->lightIndex[1] == it2->lightIndex[1]) {
                // Share light

                if (it->largeArmor != it2->largeArmor) {  // one small one large, prioritize small
                    return it->largeArmor ? it : it2;
                }

                // Remove the one that has lights more nonparallel
                return (it->lightAngleDiff > it2->lightAngleDiff) ? it : it2;
            }
        }
    }
    return acceptedArmors.end();  // nothing to remove
}

void ArmorDetector::drawRotatedRect(cv::Mat &img, const cv::RotatedRect &rect, const cv::Scalar &boarderColor) {
    cv::Point2f vertices[4];
    rect.points(vertices);
    for (int i = 0; i < 4; i++) {
        cv::line(img, vertices[i], vertices[(i + 1) % 4], boarderColor, 2);
    }
}

void ArmorDetector::canonicalizeRotatedRect(cv::RotatedRect &rect) {
    // https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned/21427814#21427814

    if (rect.size.width > rect.size.height) {
        std::swap(rect.size.width, rect.size.height);
        rect.angle += 90;
    }
    if (rect.angle < 0) {
        rect.angle += 180;
    } else if (rect.angle >= 180) {
        rect.angle -= 180;
    }
}

}