//
//  ViewController.m
//  test
//
//  Created by 林家豪 on 2018/3/6.
//  Copyright © 2018年 kaka lin. All rights reserved.
//
#import <AssertMacros.h>
#import <AssetsLibrary/AssetsLibrary.h>
#import <CoreImage/CoreImage.h>
#import <ImageIO/ImageIO.h>
#import "ViewController.h"

#include <sys/time.h>

#include "tensorflow_utils.h"

using tensorflow::uint8;

// If you have your own model, modify this to the file name, and make sure
// you've added the file to your app resources too.
static NSString *model_file_name = @"frozen_inference_graph";
static NSString *model_file_type = @"pb";

// This controls whether we'll be loading a plain GraphDef proto, or a
// file created by the convert_graphdef_memmapped_format utility that wraps a
// GraphDef and parameter file that can be mapped into memory from file to
// reduce overall memory usage.
const bool model_uses_memory_mapping = false;

// If you have your own model, point this to the labels file.
static NSString* labels_file_name = @"coco_classes";
static NSString* labels_file_type = @"txt";

// These dimensions need to match those the model was trained with.
const int wanted_input_width = 224;
const int wanted_input_height = 224;
const int wanted_input_channels = 3;
const int max_boxes = 10;
const float min_score_thres = 0.5f; // 0.5

const float input_mean = 117.0f;
const float input_std = 1.0f;
//const std::string input_layer_name = "input";
//const std::string output_layer_name = "softmax1";

@interface ViewController (InternalModel)
- (void)setupAVCapture;
- (void)teardownAVCapture;
@end

@implementation ViewController
- (void)viewDidLoad
{
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    //square = [UIImage imageNamed:@"squarePNG"];
    //synth = [[AVSpeechSynthesizer alloc] init];
    labelLayers = [[NSMutableArray alloc] init];
    oldPredictionValues = [[NSMutableDictionary alloc] init];
    
    NSLog(@"====================== Load Model ======================");
    tensorflow::Status load_status;
    if (model_uses_memory_mapping) {
        load_status = LoadMemoryMappedModel(model_file_name, model_file_type, &tf_session, &tf_memmapped_env);
    } else {
        load_status = LoadModel(model_file_name, model_file_type, &tf_session);
    }
    if (!load_status.ok()) {
        LOG(FATAL) << "Couldn't load model: " << load_status;
    }
    
    NSLog(@"====================== Load Label ======================");
    tensorflow::Status labels_status =
    LoadLabels(labels_file_name, labels_file_type, &labels);
    if (!labels_status.ok()) {
        LOG(FATAL) << "Couldn't load labels: " << labels_status;
    }
    
    [self setupAVCapture];
}


- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


- (void)setupAVCapture
{
    NSError *error = nil;
    
    // 1. create the capture session
    session = [AVCaptureSession new];
    
    // 2. 設定畫面大小
    if ([[UIDevice currentDevice] userInterfaceIdiom] == UIUserInterfaceIdiomPhone) {
        //[session setSessionPreset:AVCaptureSessionPreset640x480];
        session.sessionPreset = AVCaptureSessionPreset640x480;
    }
    else {
        //[session setSessionPreset:AVCaptureSessionPresetPhoto];
        session.sessionPreset =AVCaptureSessionPresetPhoto;
    }
    
    // 3. creat device
    AVCaptureDevice *device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    
    // 4. device input (將Device設成輸入端，可以想成輸入為Camera擷取的影像，輸出為我們設定的ImageView)
    AVCaptureDeviceInput *deviceInput =
    [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
    
    if (error != nil) {
        NSLog(@"Failed to initialize AVCaptureDeviceInput. Note: This app doesn't work with simulator");
        assert(NO);
    }
    
    // 5. connect the device input
    if ([session canAddInput:deviceInput]) [session addInput:deviceInput];
    
    /* --------------- 至此已經可以成功擷取Camera的影像，只是缺少輸出沒辦法呈現出來 --------------- */
    
    // 6. create video data output
    videoDataOutput = [AVCaptureVideoDataOutput new];
    
    // 7. 設定輸出端的像素（Pixel）格式化，包含透明度的32位元
    //    CoreImage wants BGRA pixel format
    NSDictionary* rgbOutputSettings =
    [NSDictionary dictionaryWithObject:[NSNumber numberWithInt:kCMPixelFormat_32BGRA]
                                forKey:(id)kCVPixelBufferPixelFormatTypeKey];
    
    //[videoDataOutput setVideoSettings:rgbOutputSettings];
    videoDataOutput.videoSettings = rgbOutputSettings;
    //[videoDataOutput setAlwaysDiscardsLateVideoFrames:YES];
    videoDataOutput.alwaysDiscardsLateVideoFrames = YES;
    
    // 8. create the dispatch queue for handling capture session delegate method calls
    //    對輸出端的queue做設定
    videoDataOutputQueue = dispatch_queue_create("VideoDataOutputQueue", DISPATCH_QUEUE_SERIAL);
    [videoDataOutput setSampleBufferDelegate:self queue:videoDataOutputQueue];
    
    // 9. connect the data output
    if ([session canAddOutput:videoDataOutput]) [session addOutput:videoDataOutput];
    [[videoDataOutput connectionWithMediaType:AVMediaTypeVideo] setEnabled:YES];
    
    // 補充： AVCaptureVideoPreviewLayer是CALayer的子類，可被用於自動顯示相機產生的即時圖像
    previewLayer = [[AVCaptureVideoPreviewLayer alloc] initWithSession:session];
    //[previewLayer setBackgroundColor:[[UIColor blackColor] CGColor]];
    previewLayer.backgroundColor = UIColor.blackColor.CGColor; // 更改 view 背景的顏色
    [previewLayer setVideoGravity:AVLayerVideoGravityResizeAspect]; // 將這個圖層屬性設定為resize
    //CALayer *rootLayer = [previewView layer];
    CALayer *rootLayer = previewView.layer;
    [rootLayer setMasksToBounds:YES]; // 將超過邊筐外的sublayer裁切掉
    [previewLayer setFrame:[rootLayer bounds]];
    [rootLayer addSublayer:previewLayer];
    
    // 10. start everything
    [session startRunning];
    
    if (error) {
        NSString *title = [NSString stringWithFormat:@"Failed with error %d", (int)[error code]];
        UIAlertController* alertController =
        [UIAlertController alertControllerWithTitle:title
                                            message:[error localizedDescription]
                                     preferredStyle:UIAlertControllerStyleAlert];
        UIAlertAction *dismiss =
        [UIAlertAction actionWithTitle:@"Dismiss" style:UIAlertActionStyleDefault handler:nil];
        [alertController addAction:dismiss];
        [self presentViewController:alertController animated:YES completion:nil];
        [self teardownAVCapture];
    }
}

- (void)teardownAVCapture
{
    //[stillImageOutput removeObserver:self forKeyPath:@"isCapturingStillImage"];
    [previewLayer removeFromSuperlayer];
}

// Provides the delegate a captured image in a processed format (such as JPEG)
- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection
{
    CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    // CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    // Core Foundation必須使用CFRetain和CFRelease來進行記憶體管理
    CFRetain(pixelBuffer);
    [self runCNNOnFrame:pixelBuffer];
    CFRelease(pixelBuffer);
}

- (void)runCNNOnFrame:(CVPixelBufferRef)pixelBuffer
{
    assert(pixelBuffer != NULL);
    
    // OSType: 通常作為一種4位元組的類別標示名被使用在Mac OS里。
    OSType sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
    
    int doReverseChannels;
    if (kCVPixelFormatType_32ARGB == sourcePixelFormat) {
        doReverseChannels = 1;
    } else if (kCVPixelFormatType_32BGRA == sourcePixelFormat) {
        doReverseChannels = 0;
    } else {
        assert(false);  // Unknown source format
    }
    
    tensorflow::Tensor image_tensor(
        tensorflow::DT_UINT8,
        tensorflow::TensorShape({1, wanted_input_height, wanted_input_width, wanted_input_channels})
    );
    auto image_tensor_mapped = image_tensor.tensor<uint8, 4>(); // image_tensor_dimension
    
    /*******************************************************************************************************/
    /*
    CIImage* ciImage = [[CIImage alloc] initWithCVPixelBuffer:pixelBuffer];
    
    // Create a cgImage from the frame pixels
    CIContext *context = [CIContext contextWithOptions:nil];
    CGImageRef cgImage = [context createCGImage:ciImage fromRect:ciImage.extent];
    
    const int sourceRowBytes = (int) CGImageGetBytesPerRow(cgImage);
    const int sourceHeight   = (int) CGImageGetHeight(cgImage);
    const int sourceWidth    = (int) CGImageGetWidth(cgImage);
    const int srcChannels = (int) sourceRowBytes / sourceWidth;
    
    CVPixelBufferLockFlags unlockFlags = kNilOptions;
    CVPixelBufferLockBaseAddress(pixelBuffer, unlockFlags);
    
    // Scale the pixel data down, drop the alpha channel, and populate the image_tensor.
    // The source pointer iterates through the pixelBuffer and the destination pointer
    // writes pixel data into the reshaped image tensor.  Changing the GraphInputWidth and Height
    // may increase (or decrease) speed and/or accuracy.
    CFDataRef pixelData = CGDataProviderCopyData(CGImageGetDataProvider(cgImage));
    unsigned char *sourceStartAddr  = (unsigned char *) CFDataGetBytePtr(pixelData);
    
    // Scale the buffer down to the expected size and shape of the input tensor for the TF graph
    // also, drop the alpha component as the pixel format going in is BGA.
    //tensorflow::uint8 *in = sourceStartAddr;
    tensorflow::uint8 *destStartAddress = image_tensor_mapped.data();
    for (int row = 0; row < wanted_input_height; ++row)
    {
        tensorflow::uint8 *destRow = destStartAddress + (row * wanted_input_width * wanted_input_channels);
        for (int col = 0; col < wanted_input_width; ++col)
        {
            const int srcRow = (int) (col * (sourceHeight / wanted_input_height));
            const int srcCol = (int) (row * (sourceWidth  / wanted_input_width));
            tensorflow::uint8 *srcPixel = sourceStartAddr + (srcRow * sourceRowBytes) + (srcCol * srcChannels);
            tensorflow::uint8 *destPixel = destRow + (col * wanted_input_channels);
            for (int c = 0; c < wanted_input_channels; ++c)
            {
                //destPixel[c] = srcPixel[c];
                destPixel[c] = (srcPixel[c] - input_mean) / input_std;
            }
        }
    }
    */
    /*******************************************************************************************************/

    const int sourceRowBytes = (int)CVPixelBufferGetBytesPerRow(pixelBuffer);
    const int image_width = (int)CVPixelBufferGetWidth(pixelBuffer);
    const int fullHeight = (int)CVPixelBufferGetHeight(pixelBuffer);
    
    CVPixelBufferLockFlags unlockFlags = kNilOptions;
    CVPixelBufferLockBaseAddress(pixelBuffer, unlockFlags);
    
    unsigned char *sourceBaseAddr = (unsigned char *)(CVPixelBufferGetBaseAddress(pixelBuffer));
    int image_height;
    unsigned char *sourceStartAddr;
    if (fullHeight <= image_width) {
        image_height = fullHeight;
        sourceStartAddr = sourceBaseAddr;
    } else {
        image_height = image_width;
        const int marginY = ((fullHeight - image_width) / 2);
        sourceStartAddr = (sourceBaseAddr + (marginY * sourceRowBytes));
    }
    const int image_channels = 4;
    
    assert(image_channels >= wanted_input_channels);
    
    tensorflow::uint8 *in = sourceStartAddr;
    tensorflow::uint8 *out = image_tensor_mapped.data();
    for (int y = 0; y < wanted_input_height; ++y) {
        tensorflow::uint8 *out_row = out + (y * wanted_input_width * wanted_input_channels);
        for (int x = 0; x < wanted_input_width; ++x) {
            /* 用下面寫法會錯:
             * const int in_x = (y * image_width) / wanted_input_width;
             * const int in_y = (x * image_height) / wanted_input_height;
             */
            const int in_x = (int)(y * (image_width / wanted_input_width));
            const int in_y = (int)(x * (image_height / wanted_input_height));
            tensorflow::uint8 *in_pixel = in + (in_y * image_width * image_channels) + (in_x * image_channels);
            tensorflow::uint8 *out_pixel = out_row + (x * wanted_input_channels);
            for (int c = 0; c < wanted_input_channels; ++c) {
                out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
            }
        }
    }
    
    /*******************************************************************************************************/
    
    CVPixelBufferUnlockBaseAddress(pixelBuffer, unlockFlags);
    
    if (tf_session.get()) {
        std::vector<tensorflow::Tensor> outputs;
        
        tensorflow::Status run_status = tf_session->Run(
            {{"image_tensor:0", image_tensor}}, {"detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"},{}, &outputs
        );
        
        if (!run_status.ok()) {
            LOG(ERROR) << "Running model failed:" << run_status;
        } else {
            NSLog(@"=========== Non_max_suppression ==========");
            auto boxes_flat = outputs[0].flat<float>();
            tensorflow::TTypes<float>::Flat scores_flat = outputs[1].flat<float>();
            tensorflow::TTypes<float>::Flat indices_flat = outputs[2].flat<float>();
            
            // Non_max_suppression
            NSMutableArray *out_score = [[NSMutableArray alloc] init];
            NSMutableArray *out_label = [[NSMutableArray alloc] init];
            NSMutableArray *out_boxes = [[NSMutableArray alloc] init];
            for (int i = 0; i < max_boxes; ++i) {
                const float score = scores_flat(i);
                //NSLog(@"score: %f", score);
                if (score > min_score_thres) {
                    [out_score addObject:[NSNumber numberWithFloat:score]];
                    std::string label = labels[(tensorflow::StringPiece::size_type)indices_flat(i)];
                    [out_label addObject:[NSString stringWithUTF8String:label.c_str()]];
                    [out_boxes addObject:[NSArray arrayWithObjects:
                                          [NSNumber numberWithFloat:(boxes_flat(i * 4 + 0) * image_height)],
                                          [NSNumber numberWithFloat:(boxes_flat(i * 4 + 1) * image_width)],
                                          [NSNumber numberWithFloat:(boxes_flat(i * 4 + 2) * image_height)],
                                          [NSNumber numberWithFloat:(boxes_flat(i * 4 + 3) * image_width)], nil
                                          ]];
                    
                    NSLog (@"score: %f, label is :%@, boxes: %f, %f, %f, %f", score, [NSString stringWithUTF8String:label.c_str()], boxes_flat(i * 4 + 0), boxes_flat(i * 4 + 1),  boxes_flat(i * 4 + 2), boxes_flat(i * 4 + 3));
                    
                }
            }
            
            dispatch_async(dispatch_get_main_queue(), ^(void){
                [self setPredictionWithLabels: out_label
                                       scores: out_score
                                        boxes: out_boxes
                 ];
            });
        }
    }
    
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
}

- (void)dealloc
{
    [self teardownAVCapture];
}

-(void)setPredictionWithLabels:(NSArray *)out_label
                        scores:(NSArray *)out_score
                         boxes:(NSArray *)out_boxes
{
    [self removeAllLabelLayers];
    //CGRect mainScreenBounds = [[UIScreen mainScreen] bounds];
    
    for (int i = 0; i < [out_label count]; i++) {
        NSString *label = (NSString *)out_label[i];
        [self addLabelLayerWithText:[NSString stringWithFormat:@"%@ %.2f",label,[out_score[i] floatValue]]
                                top:[out_boxes[i][0] floatValue]
                               left:[out_boxes[i][1] floatValue]
                             bottom:[out_boxes[i][2] floatValue]
                              right:[out_boxes[i][3] floatValue]
                          alignment:kCAAlignmentLeft];
    }
}

- (void)removeAllLabelLayers
{
    for (CATextLayer *layer in labelLayers) {
        [layer removeFromSuperlayer];
    }
    [labelLayers removeAllObjects];
}

- (void)addLabelLayerWithText:(NSString *)text
                          top:(float)top
                         left:(float)left
                       bottom:(float)bottom
                        right:(float)right
                    alignment:(NSString *)alignment
{
    CFTypeRef font = (CFTypeRef) @"Menlo-Regular";
    const float fontSize = 15.0f;
    const float marginSizeX = 5.0f;
    const float marginSizeY = 2.0f;

    const CGRect backgroundBounds = CGRectMake(left, top, (right - left), (bottom - top));
    NSLog(@"box x:%f y:%f width:%f height:%f",left, top, (right - left), (bottom - top));
    const CGRect textBounds = CGRectMake((left + marginSizeX), (top + marginSizeY),
                                         ((right - left) - (marginSizeX * 2)), ((bottom - top) - (marginSizeY * 2)));
    
    CATextLayer *background = [CATextLayer layer];
    [background setBackgroundColor:[UIColor blackColor].CGColor];
    [background setOpacity:0.5f];
    [background setFrame:backgroundBounds];
    background.cornerRadius = 5.0f;
    
    [[self.view layer] addSublayer:background];
    [labelLayers addObject:background];
    
    CATextLayer *layer = [CATextLayer layer];
    [layer setForegroundColor:[UIColor whiteColor].CGColor];
    [layer setFrame:textBounds];
    [layer setAlignmentMode:alignment];
    [layer setWrapped:YES];
    [layer setFont:font];
    [layer setFontSize:fontSize];
    layer.contentsScale = [[UIScreen mainScreen] scale];
    [layer setString:text];
    
    [[self.view layer] addSublayer:layer];
    [labelLayers addObject:layer];
}

@end
