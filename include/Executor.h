//
// Created by liuzikai on 4/18/21.
//

#ifndef META_VISION_SOLAIS_EXECUTOR_H
#define META_VISION_SOLAIS_EXECUTOR_H

// Include as few modules as possible and use forward declarations
#include "Parameters.h"
#include "VideoSource.h"
#include "Camera.h"
#include "ImageSet.h"
#include "ArmorDetector.h"
#include "ParamSetManager.h"
#include "PositionCalculator.h"
#include "AimingSolver.h"
#include "Serial.h"
#include <thread>

namespace meta {

class Executor : protected FrameCounterBase /* we would like to rename the function */ {
public:

    explicit Executor(Camera *camera, ImageSet *imageSet, ArmorDetector *detector, ParamSetManager *paramSetManager,
                      PositionCalculator *positionCalculator, AimingSolver *aimingSolver, Serial *serial);

    /** Read-Only Components **/

    const Camera *camera() const { return camera_; }

    const ImageSet *imageSet() const { return imageSet_; }

    const VideoSource *currentInput() const { return currentInput_; };

    const ArmorDetector *detector() const { return detector_; }

    const ParamSetManager *dataManager() const { return paramSetManager_; };

    const PositionCalculator *positionCalculator() const { return positionCalculator_; }

    const AimingSolver *aimingSolver() const { return aimingSolver_; }

    const Serial *serial() const { return serial_; }

    /** Parameter Sets and Image Lists Control **/

    void reloadLists();

    int switchImageSet(const std::string &path);

    void switchParamSet(const std::string &paramSetName);

    void saveAndApplyParams(const ParamSet &p);

    const ParamSet &getCurrentParams() const { return params; }

    std::string captureImageFromCamera();

    /** Execution **/

    enum Action {
        NONE,
        REAL_TIME_DETECTION,
        SINGLE_IMAGE_DETECTION
    };

    Action getCurrentAction() const { return curAction; }

    void stop();

    bool startRealTimeDetection();

    /**
     * Run detection on a single image. This operation is blocking and sets current action to SINGLE_IMAGE_DETECTION,
     * without resetting to NONE after completion. TCP socket can sent the result based on getCurrentAction() != NONE,
     * but should call stop() if the current action is SINGLE_IMAGE_DETECTION.
     * @param imageName
     * @return
     */
    bool startSingleImageDetection(const std::string &imageName);

    bool startImageSetDetection();

    /** Statistics and Output **/

    unsigned int fetchAndClearExecutorFrameCounter() { return FrameCounterBase::fetchAndClearFrameCounter(); }

    unsigned int fetchAndClearInputFrameCounter();

    unsigned int fetchAndClearSerialFrameCounter() { return serial_ ? serial_->fetchAndClearFrameCounter() : 0; }

    std::mutex &detectorOutputMutex() { return detector_->outputMutex; }

    std::mutex armorsOutputMutex;
    const std::vector<AimingSolver::ArmorInfo> &armorsOutput() const { return armorsOutput_; }

private:

    Camera *camera_;
    ImageSet *imageSet_;
    ArmorDetector *detector_;
    ParamSetManager *paramSetManager_;
    PositionCalculator *positionCalculator_;
    AimingSolver *aimingSolver_;
    Serial *serial_;

    VideoSource *currentInput_ = nullptr;

    ParamSet params;

    Action curAction = NONE;

    std::thread *th = nullptr;
    bool threadShouldExit = false;

    std::vector<AimingSolver::ArmorInfo> armorsOutput_;

    void applyParams(const ParamSet &p);

    void runStreamingDetection(VideoSource *source);

};

}


#endif //META_VISION_SOLAIS_EXECUTOR_H
