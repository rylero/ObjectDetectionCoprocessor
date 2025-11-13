#include "rfdetr_inference.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model> <path_to_coco_labels> [--segmentation]" << std::endl;
        return 1;
    }

    const std::filesystem::path model_path = argv[1];
    const std::filesystem::path label_file_path = argv[2];
    
    bool use_segmentation = false;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--segmentation") == 0) {
            use_segmentation = true;
        }
    }

    try {
        Config config;
        config.resolution = 0;
        config.model_type = use_segmentation ? ModelType::SEGMENTATION : ModelType::DETECTION;
        config.max_detections = 300;
        config.mask_threshold = 0.0f;
        
        RFDETRInference inference(model_path, label_file_path, config);

        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not access the camera." << std::endl;
            return 1;
        }
        
        std::cout << "Camera accessed successfully! Press ESC to quit." << std::endl;
        
        cv::Mat frame;
        
        // FPS calculation variables
        auto prev_time = std::chrono::high_resolution_clock::now();
        double fps = 0.0;
        const double fps_alpha = 0.1;
        
        while (true) {
            // Start timing for this frame
            auto start_time = std::chrono::high_resolution_clock::now();
            
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "Error: Empty frame captured" << std::endl;
                break;
            }
            
            int orig_h, orig_w;
            
            // Preprocess and run inference
            std::vector<float> input_data = inference.preprocess_image(frame, orig_h, orig_w);
            inference.run_inference(input_data);
            
            // Post-process outputs
            std::vector<float> scores;
            std::vector<int> class_ids;
            std::vector<std::vector<float>> boxes;
            std::vector<cv::Mat> masks;
            
            const float scale_w = static_cast<float>(orig_w) / inference.get_resolution();
            const float scale_h = static_cast<float>(orig_h) / inference.get_resolution();
            
            if (use_segmentation) {
                inference.postprocess_segmentation_outputs(scale_w, scale_h, orig_h, orig_w, 
                                                           scores, class_ids, boxes, masks);
                inference.draw_segmentation_masks(frame, boxes, class_ids, scores, masks);
            } else {
                inference.postprocess_outputs(scale_w, scale_h, scores, class_ids, boxes);
                inference.draw_detections(frame, boxes, class_ids, scores);
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end_time - start_time;
            double current_fps = 1.0 / elapsed.count();

            // Exponential moving average for smoother FPS display
            fps = fps_alpha * current_fps + (1.0 - fps_alpha) * fps;

            // Draw smoothed FPS
            std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
            cv::putText(frame, fps_text, cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, 
                        cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            
            // Display the frame
            cv::imshow("RFDETR Live Detection", frame);
            
            char key = static_cast<char>(cv::waitKey(1));
            if (key == 27) {
                break;
            }
        }
        
        cap.release();
        cv::destroyAllWindows();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
