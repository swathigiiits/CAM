# quick_rotate_fix.py
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId
import cv2
import os

vrs_path = r"E:\aria_work\Sree_5.vrs"
output_video = "rgb_corrected.mp4"

print("Extracting and correcting RGB video...")

provider = data_provider.create_vrs_data_provider(vrs_path)
rgb_id = StreamId(214, 1)

frame_count = provider.get_num_data(rgb_id)
print(f"Total frames: {frame_count}")

if frame_count > 0:
    # Get first frame
    first = provider.get_image_data_by_index(rgb_id, 0)
    original = first[0].to_numpy_array()
    h, w = original.shape[:2]
    
    # Rotate -90°: swap dimensions
    corrected_h, corrected_w = w, h  # After rotation
    
    # Create video with corrected dimensions
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (corrected_w, corrected_h))
    
    print(f"Original: {w}x{h}")
    print(f"After -90° rotation: {corrected_w}x{corrected_h}")
    
    for i in range(frame_count):
        data = provider.get_image_data_by_index(rgb_id, i)
        if data:
            frame = data[0].to_numpy_array()
            
            # ROTATE -90° (clockwise)
            # Method 1: Fastest
            rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            # Alternative if Method 1 doesn't work:
            # rotated = cv2.transpose(frame)
            # rotated = cv2.flip(rotated, 0)  # Vertical flip
            
            out.write(rotated)
        
        if i % 500 == 0:
            print(f"  Frame {i}/{frame_count}")
    
    out.release()
    
    print(f"\n✅ Saved: {output_video}")
    print(f"   Size: {os.path.getsize(output_video)/(1024*1024):.1f} MB")
    print(f"   Duration: {frame_count/20:.1f} seconds ({frame_count/20/60:.1f} minutes)")