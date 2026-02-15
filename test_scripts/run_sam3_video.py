import time
import os
import cv2
from tqdm import tqdm

#################################### For Video ####################################
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import render_masklet_frame, generate_colors

# Generate colors for visualization
COLORS = generate_colors(n_colors=128, n_samples=5000)


def save_video_with_overlays(
    video_path, outputs_per_frame, output_video_path, alpha=0.5, fps=10
):
    """
    Save a video with masks and bounding boxes overlaid on each frame.

    Args:
        video_path: str - Path to the original video
        outputs_per_frame: dict - {frame_idx: outputs_dict}
        output_video_path: str - Path to save the output video
        alpha: float - Mask overlay transparency (0-1)
        fps: int - Frames per second for output video
    """
    # Open video to get properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f"Video info: {width}x{height} @ {original_fps}fps, {total_frames} frames")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise ValueError(f"Could not open video writer for: {output_video_path}")

    # Sort frames by index
    sorted_frames = sorted(outputs_per_frame.keys())

    print(f"Processing {len(sorted_frames)} frames with overlays...")

    for frame_idx in tqdm(sorted_frames, desc="Generating video"):
        outputs = outputs_per_frame[frame_idx]

        # Read the original frame from video
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Warning: Could not read frame {frame_idx}")
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply overlays
        overlay = render_masklet_frame(
            frame_rgb, outputs, frame_idx=frame_idx, alpha=alpha
        )

        # Convert back to BGR for video writer
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        writer.write(overlay_bgr)

    writer.release()
    print(f"✅ Saved overlaid video to: {output_video_path}")
    print(f"   - Total frames: {len(sorted_frames)}")
    print(f"   - Resolution: {width}x{height}")
    print(f"   - FPS: {fps}")


def propagate_in_video(predictor, session_id):
    """
    Propagate prompts through the entire video and collect outputs per frame.
    """
    outputs_per_frame = {}

    print("=" * 60)
    print("PROPAGATING: Processing all video frames...")
    print("=" * 60)

    t_start = time.time()
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    t_end = time.time()
    print(f"TIMING: Propagation completed in {(t_end - t_start) * 1000:.2f} ms")
    print(f"   - Frames processed: {len(outputs_per_frame)}")

    return outputs_per_frame


def main():
    # Configuration
    video_path = "/app/sam3/assets/videos/bedroom.mp4"
    prompt = "childs"  # Text prompt to detect objects
    output_video_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "bedroom_result.mp4"
    )
    alpha = 0.5
    fps = 10

    print("=" * 60)
    print("SAM3 VIDEO INFERENCE")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Prompt: '{prompt}'")
    print(f"Output: {output_video_path}")
    print("=" * 60)

    # Load the video predictor
    print("Loading SAM3 video predictor...")
    t1 = time.time()
    video_predictor = build_sam3_video_predictor()
    t2 = time.time()
    print(f"TIMING: Model loading: {(t2 - t1) * 1000:.2f} ms")

    # Start a session
    print("\nStarting video session...")
    t3 = time.time()
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    t4 = time.time()
    session_id = response["session_id"]
    print(f"TIMING: Session start: {(t4 - t3) * 1000:.2f} ms")
    print(f"Session ID: {session_id}")

    # Add text prompt on frame 0
    print(f"\nAdding text prompt: '{prompt}' on frame 0...")
    t5 = time.time()
    response = video_predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=prompt,
        )
    )
    t6 = time.time()
    print(f"TIMING: Prompt processing: {(t6 - t5) * 1000:.2f} ms")

    # Get initial outputs
    initial_outputs = response["outputs"]
    print("Initial detection on frame 0:")
    print(f"   - Objects detected: {len(initial_outputs.get('out_probs', []))}")
    if len(initial_outputs.get("out_probs", [])) > 0:
        for i, prob in enumerate(initial_outputs["out_probs"]):
            print(f"   - Object {i}: score={prob:.3f}")

    # Propagate through video
    outputs_per_frame = propagate_in_video(video_predictor, session_id)

    # Save video with overlays
    print("\n" + "=" * 60)
    print("SAVING: Creating overlaid video...")
    print("=" * 60)

    t7 = time.time()
    save_video_with_overlays(
        video_path=video_path,
        outputs_per_frame=outputs_per_frame,
        output_video_path=output_video_path,
        alpha=alpha,
        fps=fps,
    )
    t8 = time.time()
    print(f"TIMING: Video generation: {(t8 - t7) * 1000:.2f} ms")

    # Close the session
    print("\nClosing session...")
    video_predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )

    # Final summary
    total_time = t8 - t1
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  - Video: {video_path}")
    print(f"  - Prompt: '{prompt}'")
    print(f"  - Frames processed: {len(outputs_per_frame)}")
    print(f"  - Total time: {total_time * 1000:.2f} ms ({total_time:.2f} s)")
    print(f"  - Result saved to: {output_video_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
