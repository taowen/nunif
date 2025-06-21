import cv2
import argparse
import os

def check_mp4_moov_atom(video_path):
    """
    Parses the top-level atoms of an MP4 file to check for the 'moov' atom.
    A missing 'moov' atom is a common reason for unplayable MP4 files.
    Returns True if found, False if not found, None on I/O error.
    """
    try:
        with open(video_path, 'rb') as f:
            file_size = os.path.getsize(video_path)
            pos = 0
            while pos < file_size:
                f.seek(pos)
                
                size_bytes = f.read(4)
                if len(size_bytes) < 4:
                    break
                size = int.from_bytes(size_bytes, 'big')

                type_bytes = f.read(4)
                if len(type_bytes) < 4:
                    break
                
                if type_bytes == b'moov':
                    return True
                
                atom_header_size = 8
                if size == 1:
                    size_bytes = f.read(8)
                    if len(size_bytes) < 8:
                        break
                    size = int.from_bytes(size_bytes, 'big')
                    atom_header_size = 16
                elif size == 0:
                    return False

                if size < atom_header_size:
                    return False
                
                pos += size
            return False
    except (IOError, OSError):
        return None

def analyze_video(video_path):
    """
    Analyzes a video file and prints its properties.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        file_ext = os.path.splitext(video_path)[1].lower()
        if file_ext in ['.mp4', '.mov', '.m4a', '.3gp', '.3g2', '.mj2']:
            has_moov = check_mp4_moov_atom(video_path)
            if has_moov is False:
                print("\n[!] DIAGNOSIS: The 'moov' atom was not found in the file.")
                print("    This is a common issue with MP4/MOV files that were not finalized correctly.")
                print("    The file is likely incomplete or corrupt. The program that created it may have crashed or exited before writing the file trailer.")
            elif has_moov is None:
                print("\n[!] DIAGNOSIS: Could not read the file to check for MP4 structure. It might be a permission issue.")
        return None

    print(f"\n--- Analyzing {os.path.basename(video_path)} ---")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    try:
        fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    except:
        fourcc_str = "N/A"

    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total Frames (reported by header): {frame_count}")
    print(f"Codec (FourCC): {fourcc_str} ({hex(fourcc_int)})")

    # Try to read all frames to see if the stream is valid
    read_frames = 0
    first_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if read_frames == 0:
            first_frame = frame
        read_frames += 1

    print(f"Successfully read frames: {read_frames}")

    if first_frame is not None:
        first_frame_path = f"py_debug_frame_{os.path.basename(video_path)}.png"
        cv2.imwrite(first_frame_path, first_frame)
        print(f"Saved the first valid frame to {first_frame_path}")
    else:
        print("Could not read any frames from the video stream.")

    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "reported_frames": frame_count,
        "read_frames": read_frames,
        "codec": fourcc_str,
    }

def main():
    parser = argparse.ArgumentParser(description="Diagnose video file playback issues using OpenCV.")
    parser.add_argument("processed_video", help="Path to the processed video file that cannot be played.")
    parser.add_argument("-o", "--original_video", help="Optional: Path to the original video file for comparison.", default=None)

    args = parser.parse_args()

    processed_stats = analyze_video(args.processed_video)

    if args.original_video:
        original_stats = analyze_video(args.original_video)
        if original_stats and processed_stats:
            print("\n--- Comparison ---")
            if original_stats["width"] != processed_stats["width"] or original_stats["height"] != processed_stats["height"]:
                print(f"[*] WARNING: Resolution mismatch!")
                print(f"    Original:  {original_stats['width']}x{original_stats['height']}")
                print(f"    Processed: {processed_stats['width']}x{processed_stats['height']}")
            else:
                print("[+] Resolution: OK")

            if abs(original_stats["fps"] - processed_stats["fps"]) > 0.1:
                print(f"[*] WARNING: FPS mismatch!")
                print(f"    Original:  {original_stats['fps']:.2f}")
                print(f"    Processed: {processed_stats['fps']:.2f}")
            else:
                print("[+] FPS: OK")

            if processed_stats["read_frames"] == 0:
                 print("[!] CRITICAL: No frames could be read from the processed video. The file is likely corrupt or has an unsupported format/codec.")
            elif abs(original_stats["read_frames"] - processed_stats["read_frames"]) > 5: # Allow small difference
                print(f"[*] WARNING: Frame count mismatch (based on readable frames)!")
                print(f"    Original:  {original_stats['read_frames']}")
                print(f"    Processed: {processed_stats['read_frames']}")
            else:
                print("[+] Frame count: OK")

if __name__ == "__main__":
    main() 