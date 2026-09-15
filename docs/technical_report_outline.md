# Technical Report Outline

## 1. System Architecture
- Video Player Process
- Fast Algorithm Process
- Slow Algorithm Process
- Communication and synchronization

## 2. Camera Calibration
- Checkerboard setup
- Camera matrix
- Distortion coefficients
- Reprojection error

## 3. Slow Detection
- Preprocessing
- A4 candidate extraction
- Quadrilateral validation
- Perspective rectification
- X verification
- Confidence

## 4. Fast Tracking
- Feature selection
- Lucas-Kanade optical flow
- RANSAC homography
- Corner propagation
- Tracking confidence

## 5. Pose Estimation
- A4 world coordinates
- solvePnP
- Position and distance
- Roll, pitch, yaw
- Reprojection error

## 6. Tracking Loss and Recovery
- Failure criteria
- Re-detection trigger
- State machine

## 7. Multiprocessing
- Queues
- Timestamps / frame IDs
- Dropping stale frames
- Handling delayed Slow results

## 8. Test Videos
- Direct approach
- Diagonal movement
- Side-to-side
- Height changes
- Sharp angles / rotation
- Far target / entry-exit from FOV

## 9. Performance Metrics
- FPS / processing time
- Confidence
- Reprojection error
- Number of tracking points
- Recovery time

## 10. Limitations and Future Improvements
