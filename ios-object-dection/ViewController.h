//
//  ViewController.h
//  ios-object-dection
//
//  Created by 林家豪 on 2018/3/9.
//  Copyright © 2018年 kaka lin. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <AVFoundation/AVFoundation.h>

#include <memory>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/memmapped_file_system.h"

@interface ViewController : UIViewController <AVCaptureVideoDataOutputSampleBufferDelegate, UIGestureRecognizerDelegate> {
    // View layer
    IBOutlet UIView *previewView;
    AVCaptureVideoPreviewLayer *previewLayer;
    
    NSMutableArray *labelLayers;
    
    // AVCapture(Camera)
    AVCaptureSession *session;
    AVCaptureVideoDataOutput *videoDataOutput;
    dispatch_queue_t videoDataOutputQueue;
    
    // tensorflow
    std::unique_ptr<tensorflow::Session> tf_session;
    std::unique_ptr<tensorflow::MemmappedEnv> tf_memmapped_env;
    std::vector<std::string> labels;
    
    NSMutableDictionary *oldPredictionValues;
    
    AVSpeechSynthesizer *synth;
    UIImage *square;
}

@property(strong, nonatomic) CATextLayer* predictionTextLayer;

@end
