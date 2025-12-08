import cv2

def detect_available_cameras(max_cams=20):
    """Try to open camera IDs and return a list of working ones."""
    available = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available

def select_camera(available_ids):
    """Let user preview and select from available cameras - EXACT max localizer logic"""
    print("Available camera IDs:", available_ids)
    for cam_id in available_ids:
        cap = cv2.VideoCapture(cam_id)
        print(f"Showing preview for camera ID {cam_id} (press any key to continue, or ESC to select this one)...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, f"PREVIEW OF CAMERA {cam_id}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Press ESC to SELECT this camera", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Press any key to SKIP this camera", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(f"Camera ID {cam_id}", frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return cam_id
            elif key != -1:
                break
        cap.release()
        cv2.destroyAllWindows()
    return available_ids[0] if available_ids else None

def stream_selected_camera(cam_id):
    """Stream the selected camera with quit option"""
    print(f"Streaming from camera {cam_id}. Press 'q' to quit.")
    
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Failed to open camera {cam_id}")
        return
    
    # Set camera properties like max localizer
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7.0)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Add info overlay
        cv2.putText(frame, f"Camera ID: {cam_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Selected Camera Stream", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Camera Streamer - Max Localizer Style")
    print("=" * 40)
    
    # Detect available cameras
    available = detect_available_cameras()
    if not available:
        print("No cameras found!")
        return
    
    # Select camera using max localizer logic
    cam_id = select_camera(available)
    if cam_id is None:
        print("No camera selected!")
        return
    
    # Stream the selected camera
    stream_selected_camera(cam_id)
    print("Camera streaming stopped.")

if __name__ == "__main__":
    main()
