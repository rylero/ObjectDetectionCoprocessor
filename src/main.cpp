#include "rfdetr_inference.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>

// Structure to hold object position and orientation data
struct ObjectPosition {
    std::string class_name;
    float distance;        // in feet
    float lateral_offset;  // in feet (positive = right, negative = left)
    float angle_deg;       // signed coral yaw estimate in degrees
};

// Function to calculate world position from pixel coordinates
ObjectPosition calculate_position(float bottom_center_x, float bottom_y, 
                                  const cv::Mat& camera_matrix,
                                  float camera_height_ft = 2.0f) {
    // Extract camera intrinsics
    float fx = static_cast<float>(camera_matrix.at<double>(0, 0));
    float fy = static_cast<float>(camera_matrix.at<double>(1, 1));
    float cx = static_cast<float>(camera_matrix.at<double>(0, 2));
    float cy = static_cast<float>(camera_matrix.at<double>(1, 2));
    
    // Convert camera height to meters for calculation
    float camera_height_m = camera_height_ft * 0.3048f;
    
    // Normalize pixel coordinates
    float x_normalized = (bottom_center_x - cx) / fx;
    float y_normalized = (bottom_y - cy) / fy;
    
    // Ray direction in camera coordinates (simple pinhole model)
    float ray_x = x_normalized;
    float ray_y = y_normalized;
    float ray_z = 1.0f;

    // Avoid division by zero near the horizon
    if (std::fabs(ray_y) < 1e-6f) {
        ray_y = (ray_y >= 0.0f ? 1e-6f : -1e-6f);
    }
    
    // Ground plane intersection:
    // Use a positive scale factor so that objects below the center (y_normalized > 0)
    // produce positive depth and lateral offset sign consistent with image x.
    float t = camera_height_m / ray_y;
    
    // 3D point on ground in camera coordinates
    float ground_x = ray_x * t;  // lateral (left/right, same sign as image x)
    float ground_z = ray_z * t;  // depth (forward)
    
    // Ensure forward distance is positive
    if (ground_z < 0.0f) {
        ground_z = -ground_z;
    }
    
    // Convert to feet
    float distance_ft = ground_z / 0.3048f;
    float lateral_offset_ft = ground_x / 0.3048f;
    
    ObjectPosition pos;
    pos.distance = distance_ft;
    pos.lateral_offset = lateral_offset_ft;
    pos.angle_deg = 0.0f;  // will be filled in later
    return pos;
}

// Estimate unsigned coral angle (0..90 deg) from bounding-box aspect ratio
float estimate_coral_angle_magnitude(float width, float height) {
    float long_side  = std::max(width, height);
    float short_side = std::max(1.0f, std::min(width, height));  // avoid div-by-zero
    float aspect = long_side / short_side;

    const float ASPECT_FRONT = 1.56f; // tune with field data if needed
    const float ASPECT_SIDE  = 2.90f; // near L/D, can be tuned

    float t = (aspect - ASPECT_FRONT) / (ASPECT_SIDE - ASPECT_FRONT);
    t = std::clamp(t, 0.0f, 1.0f);

    return 90.0f * t;
}

// Use diagonal corner brightness to choose sign of angle
float disambiguate_coral_angle_sign(const cv::Mat& frame,
                                    const std::vector<float>& box,
                                    float angle_mag_deg) {
    if (angle_mag_deg < 5.0f) {
        return 0.0f;
    }
    if (angle_mag_deg > 85.0f) {
        return 90.0f;
    }

    int x1 = std::max(0, std::min(frame.cols - 1, static_cast<int>(std::round(box[0]))));
    int y1 = std::max(0, std::min(frame.rows - 1, static_cast<int>(std::round(box[1]))));
    int x2 = std::max(0, std::min(frame.cols - 1, static_cast<int>(std::round(box[2]))));
    int y2 = std::max(0, std::min(frame.rows - 1, static_cast<int>(std::round(box[3]))));

    if (x2 <= x1 + 2 || y2 <= y1 + 2) {
        return angle_mag_deg;
    }

    cv::Rect roi_rect(x1, y1, x2 - x1, y2 - y1);
    cv::Mat roi = frame(roi_rect);

    cv::Mat gray;
    if (roi.channels() == 3) {
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = roi;
    }

    int patch_size = std::max(4, std::min(gray.cols, gray.rows) / 5);

    auto mean_patch = [&](int px, int py) -> double {
        px = std::max(0, std::min(gray.cols - patch_size, px));
        py = std::max(0, std::min(gray.rows - patch_size, py));
        cv::Rect pr(px, py, patch_size, patch_size);
        cv::Scalar m = cv::mean(gray(pr));
        return m[0];
    };

    double tl = mean_patch(0, 0);
    double tr = mean_patch(gray.cols - patch_size, 0);
    double bl = mean_patch(0, gray.rows - patch_size);
    double br = mean_patch(gray.cols - patch_size, gray.rows - patch_size);

    double diag1 = tl + br; // TL + BR
    double diag2 = tr + bl; // TR + BL

    float sign = (diag1 >= diag2) ? 1.0f : -1.0f;
    return angle_mag_deg * sign;
}

// Function to draw minimap
void draw_minimap(cv::Mat& frame, const std::vector<ObjectPosition>& positions,
                  const std::vector<int>& class_ids,
                  const RFDETRInference& inference) {
    int map_width = 300;
    int map_height = 300;
    int margin = 20;
    
    cv::Rect map_rect(margin, frame.rows - map_height - margin, map_width, map_height);
    if (map_rect.y < 0) {
        return;
    }

    cv::Mat minimap = cv::Mat::zeros(map_height, map_width, CV_8UC3);
    minimap.setTo(cv::Scalar(40, 40, 40));
    
    cv::Scalar grid_color(80, 80, 80);
    for (int i = 0; i <= 10; i++) {
        int x = i * map_width / 10;
        cv::line(minimap, cv::Point(x, 0), cv::Point(x, map_height), grid_color, 1);
        int y = i * map_height / 10;
        cv::line(minimap, cv::Point(0, y), cv::Point(map_width, y), grid_color, 1);
    }
    
    int center_x = map_width / 2;
    int bottom_y = map_height - 10;
    cv::line(minimap, cv::Point(center_x, bottom_y), cv::Point(center_x, 0), 
             cv::Scalar(255, 255, 255), 2);
    cv::line(minimap, cv::Point(0, bottom_y), cv::Point(map_width, bottom_y), 
             cv::Scalar(255, 255, 255), 2);
    
    float scale = map_height / 15.0f;
    
    cv::circle(minimap, cv::Point(center_x, bottom_y), 5, cv::Scalar(0, 255, 0), -1);
    cv::putText(minimap, "Camera", cv::Point(center_x + 8, bottom_y), 
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
    
    for (size_t i = 0; i < positions.size(); i++) {
        const auto& pos = positions[i];
        
        int map_x = center_x + static_cast<int>(pos.lateral_offset * scale);
        int map_y = bottom_y - static_cast<int>(pos.distance * scale);
        
        map_x = std::max(5, std::min(map_width - 5, map_x));
        map_y = std::max(5, std::min(map_height - 5, map_y));
        
        cv::circle(minimap, cv::Point(map_x, map_y), 4, cv::Scalar(0, 0, 255), -1);
        
        std::string label = inference.coco_labels_[class_ids[i]];
        std::string distance_str = std::to_string(static_cast<int>(pos.distance)) + "ft";
        
        cv::putText(minimap, label, cv::Point(map_x + 6, map_y - 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(255, 255, 255), 1);
        cv::putText(minimap, distance_str, cv::Point(map_x + 6, map_y + 8), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(200, 200, 200), 1);
    }
    
    cv::putText(minimap, "15ft", cv::Point(5, 15), 
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    cv::putText(minimap, "0ft", cv::Point(5, map_height - 5), 
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    
    cv::Mat roi = frame(map_rect);
    cv::addWeighted(roi, 0.3, minimap, 0.7, 0, roi);
}

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

        cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 
            934.9030257550975, 0, 611.9876567314589,
            0, 938.0399426396231, 361.444437099335,
            0, 0, 1);
        
        cv::Mat dist_coeffs = (cv::Mat_<double>(5, 1) << 
            -0.011903285318003488,
            -0.05583573295105518,
            -0.0009058639291767295,
            -0.005807132897704802,
            -0.01956109607982254);

        cv::VideoCapture cap(0, cv::CAP_V4L2);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not access the camera." << std::endl;
            return 1;
        }
        
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        
        std::cout << "Camera accessed successfully! Press ESC to quit." << std::endl;
        
        cv::Mat frame, undistorted_frame;
        
        using clock = std::chrono::high_resolution_clock;

        double fps = 0.0;
        const double fps_alpha = 0.1;

        // Running EMA for profiling (ms)
        double capture_ms_ema    = 0.0;
        double undistort_ms_ema  = 0.0;
        double preprocess_ms_ema = 0.0;
        double infer_ms_ema      = 0.0;
        double post_ms_ema       = 0.0;
        double total_ms_ema      = 0.0;

        const double prof_alpha = 0.1;

        const int display_scale = 2;
        
        while (true) {
            auto t_frame_start = clock::now();

            auto t_capture_start = clock::now();
            cap >> frame;
            auto t_capture_end = clock::now();

            if (frame.empty()) {
                std::cerr << "Error: Empty frame captured" << std::endl;
                break;
            }
            
            auto t_undistort_start = clock::now();
            cv::undistort(frame, undistorted_frame, camera_matrix, dist_coeffs);
            auto t_undistort_end = clock::now();
            
            int orig_h, orig_w;
            
            auto t_preprocess_start = clock::now();
            std::vector<float> input_data = inference.preprocess_image(undistorted_frame, orig_h, orig_w);
            auto t_preprocess_end = clock::now();

            auto t_infer_start = clock::now();
            inference.run_inference(input_data);
            auto t_infer_end = clock::now();
            
            std::vector<float> scores;
            std::vector<int> class_ids;
            std::vector<std::vector<float>> boxes;
            std::vector<cv::Mat> masks;
            
            const float scale_w = static_cast<float>(orig_w) / inference.get_resolution();
            const float scale_h = static_cast<float>(orig_h) / inference.get_resolution();
            
            auto t_post_start = clock::now();
            if (use_segmentation) {
                inference.postprocess_segmentation_outputs(scale_w, scale_h, orig_h, orig_w, 
                                                           scores, class_ids, boxes, masks);
                inference.draw_segmentation_masks(undistorted_frame, boxes, class_ids, scores, masks);
            } else {
                inference.postprocess_outputs(scale_w, scale_h, scores, class_ids, boxes);
                inference.draw_detections(undistorted_frame, boxes, class_ids, scores);
            }
            
            std::vector<ObjectPosition> positions;
            positions.reserve(boxes.size());

            for (size_t i = 0; i < boxes.size(); i++) {
                float x1 = boxes[i][0];
                float y1 = boxes[i][1];
                float x2 = boxes[i][2];
                float y2 = boxes[i][3];
                
                float bottom_center_x = (x1 + x2) * 0.5f;
                float bottom_y        = y2;
                
                ObjectPosition pos = calculate_position(bottom_center_x, bottom_y, camera_matrix);
                pos.class_name = inference.coco_labels_[class_ids[i]];

                float box_w = std::max(1.0f, x2 - x1);
                float box_h = std::max(1.0f, y2 - y1);
                float angle_mag = estimate_coral_angle_magnitude(box_w, box_h);
                float angle_signed = disambiguate_coral_angle_sign(undistorted_frame, boxes[i], angle_mag);
                pos.angle_deg = angle_signed;

                positions.push_back(pos);
                
                std::string pos_text = 
                    std::to_string(static_cast<int>(pos.distance)) + "ft, " +
                    std::to_string(static_cast<int>(pos.lateral_offset)) + "ft, " +
                    std::to_string(static_cast<int>(pos.angle_deg)) + "deg";
                
                cv::putText(undistorted_frame, pos_text, 
                            cv::Point(static_cast<int>(x1), static_cast<int>(y2) + 20), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
            }
            
            draw_minimap(undistorted_frame, positions, class_ids, inference);
            auto t_post_end = clock::now();

            auto t_frame_end = clock::now();

            // Durations in ms
            auto capture_ms   = std::chrono::duration<double, std::milli>(t_capture_end   - t_capture_start).count();
            auto undistort_ms = std::chrono::duration<double, std::milli>(t_undistort_end - t_undistort_start).count();
            auto preprocess_ms= std::chrono::duration<double, std::milli>(t_preprocess_end- t_preprocess_start).count();
            auto infer_ms     = std::chrono::duration<double, std::milli>(t_infer_end     - t_infer_start).count();
            auto post_ms      = std::chrono::duration<double, std::milli>(t_post_end      - t_post_start).count();
            auto total_ms     = std::chrono::duration<double, std::milli>(t_frame_end     - t_frame_start).count();

            double current_fps = 1000.0 / std::max(1e-3, total_ms);
            fps = fps_alpha * current_fps + (1.0 - fps_alpha) * fps;

            auto ema = [&](double &ema_val, double sample) {
                if (ema_val == 0.0) ema_val = sample;
                else ema_val = prof_alpha * sample + (1.0 - prof_alpha) * ema_val;
            };
            ema(capture_ms_ema,    capture_ms);
            ema(undistort_ms_ema,  undistort_ms);
            ema(preprocess_ms_ema, preprocess_ms);
            ema(infer_ms_ema,      infer_ms);
            ema(post_ms_ema,       post_ms);
            ema(total_ms_ema,      total_ms);

            // Draw FPS and profiling overlay (top-left)
            int x = 10;
            int y = 20;
            int dy = 18;
            double font_scale = 0.5;
            int thickness = 1;
            int line_type = cv::LINE_AA;
            cv::Scalar color(0, 255, 0);

            char buf[128];

            std::snprintf(buf, sizeof(buf), "FPS: %.1f", fps);
            cv::putText(undistorted_frame, buf, cv::Point(x, y), 
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, line_type);
            y += dy;

            std::snprintf(buf, sizeof(buf), "Cap:   %.2f ms", capture_ms_ema);
            cv::putText(undistorted_frame, buf, cv::Point(x, y), 
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, line_type);
            y += dy;

            std::snprintf(buf, sizeof(buf), "Undist:%.2f ms", undistort_ms_ema);
            cv::putText(undistorted_frame, buf, cv::Point(x, y), 
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, line_type);
            y += dy;

            std::snprintf(buf, sizeof(buf), "Prep:  %.2f ms", preprocess_ms_ema);
            cv::putText(undistorted_frame, buf, cv::Point(x, y), 
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, line_type);
            y += dy;

            std::snprintf(buf, sizeof(buf), "Infer: %.2f ms", infer_ms_ema);
            cv::putText(undistorted_frame, buf, cv::Point(x, y), 
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, line_type);
            y += dy;

            std::snprintf(buf, sizeof(buf), "Post:  %.2f ms", post_ms_ema);
            cv::putText(undistorted_frame, buf, cv::Point(x, y), 
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, line_type);
            y += dy;

            std::snprintf(buf, sizeof(buf), "Total: %.2f ms", total_ms_ema);
            cv::putText(undistorted_frame, buf, cv::Point(x, y), 
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, line_type);

            // Scale up for display
            cv::Mat display_frame;
            cv::resize(undistorted_frame, display_frame, cv::Size(), display_scale, display_scale, cv::INTER_LINEAR);
            
            cv::imshow("RFDETR Live Detection", display_frame);
            
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
