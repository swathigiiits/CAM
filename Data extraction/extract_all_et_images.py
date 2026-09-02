# extract_all_et_images.py
import cv2
import numpy as np
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId
import os

def extract_and_analyze_et_images():
    """Extract ALL eye tracking images and analyze when they have data"""
    
    vrs_file = "sreevatsav-tada-p21.vrs"
    output_dir = "sree_dt"
    
    print(f"=== Extracting ALL Eye Tracking Images ===")
    
    # Open VRS
    provider = data_provider.create_vrs_data_provider(vrs_file)
    eye_id = StreamId(211, 1)
    
    # Get total count
    total_images = provider.get_num_data(eye_id)
    print(f"Total ET images: {total_images}")
    
    # Create output structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "with_data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "no_data"), exist_ok=True)
    
    print(f"\nAnalyzing images...")
    
    stats = {
        'total': 0,
        'has_data': 0,
        'no_data': 0,
        'first_data_idx': None,
        'last_data_idx': None
    }
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    num_batches = (total_images + batch_size - 1) // batch_size
    
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, total_images)
        
        print(f"\nBatch {batch+1}/{num_batches}: images {start_idx}-{end_idx-1}")
        
        for i in range(start_idx, end_idx):
            try:
                image_data = provider.get_image_data_by_index(eye_id, i)
                
                if image_data:
                    img_array = image_data[0].to_numpy_array()
                    timestamp = image_data[1].capture_timestamp_ns
                    
                    stats['total'] += 1
                    
                    # Check if image has meaningful data
                    has_data = check_if_has_eye_data(img_array)
                    
                    if has_data:
                        stats['has_data'] += 1
                        if stats['first_data_idx'] is None:
                            stats['first_data_idx'] = i
                        stats['last_data_idx'] = i
                        
                        # Save image with data
                        filename = f"et_{i:06d}_{timestamp}.jpg"
                        filepath = os.path.join(output_dir, "with_data", filename)
                        cv2.imwrite(filepath, img_array)
                        
                        # Also save enhanced version
                        enhanced = enhance_et_image(img_array)
                        enhanced_path = os.path.join(output_dir, "with_data", f"enhanced_{i:06d}.jpg")
                        cv2.imwrite(enhanced_path, enhanced)
                        
                    else:
                        stats['no_data'] += 1
                        # Only save first few no-data images
                        if stats['no_data'] <= 10:
                            filename = f"no_data_{i:06d}.jpg"
                            filepath = os.path.join(output_dir, "no_data", filename)
                            cv2.imwrite(filepath, img_array)
                    
                    # Progress update
                    if i % 500 == 0:
                        print(f"  Processed {i}/{total_images} images...")
                        
            except Exception as e:
                # Silently continue on errors
                continue
    
    # Print statistics
    print(f"\n{'='*50}")
    print("EXTRACTION COMPLETE - STATISTICS:")
    print(f"{'='*50}")
    print(f"Total images processed: {stats['total']}")
    print(f"Images WITH eye data: {stats['has_data']} ({stats['has_data']/stats['total']*100:.1f}%)")
    print(f"Images WITHOUT eye data: {stats['no_data']} ({stats['no_data']/stats['total']*100:.1f}%)")
    
    if stats['first_data_idx'] is not None:
        print(f"\nFirst image with data: #{stats['first_data_idx']}")
        print(f"Last image with data: #{stats['last_data_idx']}")
        print(f"Duration with data: images {stats['first_data_idx']}-{stats['last_data_idx']}")
        
        # Calculate time
        first_data = provider.get_image_data_by_index(eye_id, stats['first_data_idx'])
        last_data = provider.get_image_data_by_index(eye_id, stats['last_data_idx'])
        
        if first_data and last_data:
            first_time = first_data[1].capture_timestamp_ns
            last_time = last_data[1].capture_timestamp_ns
            duration_seconds = (last_time - first_time) / 1e9
            
            print(f"Time with data: {duration_seconds:.1f} seconds")
    
    print(f"\nOutput saved in: {output_dir}/")
    print(f"  - with_data/: Images that contain eye data")
    print(f"  - no_data/: First 10 images without data")
    
    return stats

def check_if_has_eye_data(img_array, threshold=20):
    """Check if image likely contains eye data"""
    
    # Simple check: if image has pixels above threshold
    max_pixel = img_array.max()
    
    # Eye images should have some bright pixels (corneal reflection, sclera)
    if max_pixel < threshold:
        return False
    
    # Check if more than just a few pixels are bright
    bright_pixels = np.sum(img_array > threshold)
    total_pixels = img_array.size
    
    # Need at least 0.5% of pixels to be bright for it to be an eye image
    return (bright_pixels / total_pixels) > 0.005

def enhance_et_image(img_array):
    """Enhance eye tracking image for better visualization"""
    
    # CLAHE works well for eye images
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    
    if len(img_array.shape) == 2:
        enhanced = clahe.apply(img_array)
    else:
        # Convert to grayscale if needed
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        enhanced = clahe.apply(gray)
    
    return enhanced

def create_et_video():
    """Create a video from eye tracking images (for visualization)"""
    
    print(f"\n=== Creating Eye Tracking Video ===")
    
    input_dir = "et_images_analysis/with_data"
    
    if not os.path.exists(input_dir):
        print(f"❌ No images with data found")
        return
    
    # Get list of images
    image_files = sorted([f for f in os.listdir(input_dir) if f.startswith('enhanced_') and f.endswith('.jpg')])
    
    if len(image_files) == 0:
        print(f"❌ No enhanced images found")
        return
    
    print(f"Found {len(image_files)} enhanced eye images")
    
    # Read first image to get dimensions
    first_img = cv2.imread(os.path.join(input_dir, image_files[0]))
    height, width = first_img.shape[:2]
    
    # Create video writer
    video_path = "secord_et.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30  # Eye tracking is 10 FPS
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    print(f"Creating video: {width}x{height} @ {fps} FPS")
    
    for i, filename in enumerate(image_files):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            out.write(img)
            
            if i % 100 == 0:
                print(f"  Added {i+1}/{len(image_files)} frames...")
    
    out.release()
    print(f"✅ Video saved: {video_path}")

if __name__ == "__main__":
    # Extract and analyze all ET images
    stats = extract_and_analyze_et_images()
    
    # Only create video if we found eye data
    if stats['has_data'] > 0:
        create_et_video()
        
        print(f"\n{'='*50}")
        print("NEXT STEPS:")
        print(f"{'='*50}")
        print("1. Check 'et_images_analysis/with_data/' folder")
        print("2. Look at 'enhanced_*.jpg' files - do they show EYES?")
        print("3. Watch 'et_eye_tracking_video.mp4' to see eye movements")
        print("\nIf you see eyes in the images, we can proceed with:")
        print("  - Blink detection")
        print("  - Pupil tracking")
        print("  - Fixation/saccade analysis")
    else:
        print(f"\n❌ NO EYE DATA FOUND in any images!")
        print("This means either:")
        print("1. Eye camera wasn't working during recording")
        print("2. Eye images are encrypted/not stored")
        print("3. You need different recording settings")