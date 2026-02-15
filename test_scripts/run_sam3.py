import time
import os
import torch
import numpy as np
from PIL import Image
import cv2

#################################### For Image ####################################
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load an image
# image_path = "/app/to_mount/asset/kcp-3-min.jpg"
image_path = "/app/sam3/assets/images/groceries.jpg"

image = Image.open(image_path)

# Time the image processing (set_image)
print("=" * 60)
print("TIMING: Starting inference timing...")
print("=" * 60)

t1 = time.time()
inference_state = processor.set_image(image)
t2 = time.time()
print(f"TIMING: Image processing (set_image): {(t2 - t1) * 1000:.2f} ms")

# Define multiple text prompts
text_prompts = ["bags", "headrest", "bottles"]
print("DEBUG: Using prompts:", text_prompts)

try:
    device = next(model.parameters()).device
except Exception:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEBUG: Model device:", device)

# Process each prompt individually, reusing the vision features
all_results = []
t3 = time.time()

for idx, prompt in enumerate(text_prompts):
    print(f"DEBUG: Processing prompt {idx + 1}/{len(text_prompts)}: '{prompt}'")

    # Create a copy of the inference state for this prompt
    # (set_text_prompt modifies state, so we need fresh copies for each prompt)
    prompt_state = inference_state.copy()

    # Run inference for this single prompt
    output = processor.set_text_prompt(prompt=prompt, state=prompt_state)

    # Store results with prompt label
    all_results.append(
        {
            "prompt": prompt,
            "masks": output["masks"],
            "boxes": output["boxes"],
            "scores": output["scores"],
        }
    )

t4 = time.time()
print(
    f"TIMING: Text prompt inference for {len(text_prompts)} prompts: {(t4 - t3) * 1000:.2f} ms"
)
print(f"TIMING: Total inference time: {(t4 - t1) * 1000:.2f} ms")

# Aggregate results from all prompts
if all_results:
    masks = (
        torch.cat(
            [
                r["masks"]
                for r in all_results
                if r["masks"] is not None and len(r["masks"]) > 0
            ],
            dim=0,
        )
        if any(r["masks"] is not None and len(r["masks"]) > 0 for r in all_results)
        else None
    )
    boxes = (
        torch.cat(
            [
                r["boxes"]
                for r in all_results
                if r["boxes"] is not None and len(r["boxes"]) > 0
            ],
            dim=0,
        )
        if any(r["boxes"] is not None and len(r["boxes"]) > 0 for r in all_results)
        else None
    )
    scores = (
        torch.cat(
            [
                r["scores"]
                for r in all_results
                if r["scores"] is not None and len(r["scores"]) > 0
            ],
            dim=0,
        )
        if any(r["scores"] is not None and len(r["scores"]) > 0 for r in all_results)
        else None
    )
else:
    masks, boxes, scores = None, None, None


def _debug_print_info(x, name: str):
    if x is None:
        print(f"DEBUG: {name} is None")
        return
    if hasattr(x, "shape"):
        try:
            print(
                f"DEBUG: {name} type={type(x)}, shape={x.shape}, dtype={getattr(x, 'dtype', None)}"
            )
            return
        except Exception:
            pass
    try:
        print(f"DEBUG: {name} type={type(x)}, len={len(x)}")
    except Exception:
        print(f"DEBUG: {name} type={type(x)}")


_debug_print_info(masks, "masks")
_debug_print_info(boxes, "boxes")
_debug_print_info(scores, "scores")


def _safe_count(x):
    """Return the number of items for x (supports list, tuple, torch/numpy tensors).
    Falls back to 0 for None or unknown types.
    """
    if x is None:
        return 0
    # Tensors and numpy arrays have a shape attribute; take first dimension
    if hasattr(x, "shape"):
        try:
            return int(x.shape[0])
        except Exception:
            pass
    try:
        return int(len(x))
    except Exception:
        return 1


masks_count = _safe_count(masks)
boxes_count = _safe_count(boxes)
scores_count = _safe_count(scores)

print("Masks count:", masks_count)
print("Boxes count:", boxes_count)
print("Scores count:", scores_count)


# Generate colors for visualization
def generate_colors(n_colors=256, n_samples=5000):
    from skimage.color import lab2rgb, rgb2lab
    from sklearn.cluster import KMeans

    np.random.seed(42)
    rgb = np.random.rand(n_samples, 3)
    lab = rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(lab)
    centers_lab = kmeans.cluster_centers_
    colors_rgb = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
    colors_rgb = np.clip(colors_rgb, 0, 1)
    return colors_rgb


COLORS = generate_colors(n_colors=128, n_samples=5000)


def save_overlaid_image(
    image, masks, boxes, scores, output_path, alpha=0.5, prompt_labels=None
):
    """
    Save an image with masks and bounding boxes overlaid.

    Args:
        image: PIL.Image - The original image
        masks: torch.Tensor or list - Binary masks (N, H, W) or list of masks
        boxes: torch.Tensor or list - Bounding boxes in XYXY format (N, 4)
        scores: torch.Tensor or list - Confidence scores (N,)
        output_path: str - Path to save the overlaid image
        alpha: float - Mask overlay transparency (0-1)
        prompt_labels: list - Optional list of prompt strings for each object
    """
    # Convert PIL image to numpy array
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = np.array(image)

    height, width = img_np.shape[:2]
    overlay = img_np.copy()

    # Ensure masks and boxes are lists/numpy arrays
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    # Handle single mask/box case
    if masks.ndim == 2:
        masks = masks[None, ...]
    if boxes.ndim == 1:
        boxes = boxes[None, ...]

    num_objects = len(scores) if scores is not None else len(masks)

    # Overlay masks
    for i in range(num_objects):
        color = COLORS[i % len(COLORS)]
        color255 = (color * 255).astype(np.uint8)

        mask = masks[i]

        # Squeeze out any extra dimensions (e.g., channel dim of size 1)
        mask = mask.squeeze()

        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

        mask_bool = mask > 0.5
        for c in range(3):
            overlay[..., c][mask_bool] = (
                alpha * color255[c] + (1 - alpha) * overlay[..., c][mask_bool]
            ).astype(np.uint8)

    # Draw bounding boxes and labels
    for i in range(num_objects):
        box = boxes[i]
        score = scores[i] if scores is not None else None
        color = COLORS[i % len(COLORS)]
        color255 = tuple(int(x * 255) for x in color)

        # Box is in XYXY format (absolute pixel coordinates or normalized)
        x1, y1, x2, y2 = box

        # Convert to integers if they are floats
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color255, 2)

        # Add label with score and prompt
        if prompt_labels and i < len(prompt_labels):
            prompt_text = prompt_labels[i]
            if score is not None:
                label = f"{prompt_text}: {score:.2f}"
            else:
                label = prompt_text
        else:
            if score is not None:
                label = f"Obj {i}: {score:.2f}"
            else:
                label = f"Obj {i}"

        cv2.putText(
            overlay,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color255,
            1,
            cv2.LINE_AA,
        )

    # Save the overlaid image
    Image.fromarray(overlay).save(output_path)
    print(f"✅ Saved overlaid image to: {output_path}")
    print(f"   - Objects detected: {num_objects}")
    print(f"   - Image size: {width}x{height}")
    print(f"   - Mask overlay alpha: {alpha}")

    return overlay


# Save the overlaid image if we have detections
if masks_count > 0:
    print("=" * 60)
    print("SAVING: Creating mask and bbox overlaid image...")
    print("=" * 60)

    # Build prompt labels for each detected object
    prompt_labels = []
    for r in all_results:
        r_count = _safe_count(r["masks"])
        for _ in range(r_count):
            prompt_labels.append(r["prompt"])

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_image_path = os.path.join(script_dir, "groceries_result.jpg")

    t5 = time.time()
    save_overlaid_image(
        image=image,
        masks=masks,
        boxes=boxes,
        scores=scores,
        output_path=output_image_path,
        alpha=0.5,
        prompt_labels=prompt_labels,
    )
    t6 = time.time()
    print(f"TIMING: Visualization generation: {(t6 - t5) * 1000:.2f} ms")
    print("=" * 60)
    print("SUMMARY:")
    print(f"  - Prompts processed: {text_prompts}")
    print(f"  - Total objects detected: {masks_count}")
    for r in all_results:
        r_count = _safe_count(r["masks"])
        print(f"    - '{r['prompt']}': {r_count} object(s)")
    print(f"  - Total time: {(t6 - t1) * 1000:.2f} ms")
    print(f"  - Result saved to: {output_image_path}")
    print("=" * 60)
else:
    print("WARNING: No objects detected. No overlaid image will be saved.")

#################################### For Video ####################################

# from sam3.model_builder import build_sam3_video_predictor

# video_predictor = build_sam3_video_predictor()
# video_path = "<YOUR_VIDEO_PATH>" # a JPEG folder or an MP4 video file
# # Start a session
# response = video_predictor.handle_request(
#     request=dict(
#         type="start_session",
#         resource_path=video_path,
#     )
# )
# response = video_predictor.handle_request(
#     request=dict(
#         type="add_prompt",
#         session_id=response["session_id"],
#         frame_index=0, # Arbitrary frame index
#         text="<YOUR_TEXT_PROMPT>",
#     )
# )
# output = response["outputs"]
