import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("/app/sam3/assets/images/groceries.jpg")
inference_state = processor.set_image(image)
# Prompt the model with text
prompt = "bags"
print("DEBUG: Using prompt:", prompt)
try:
	device = next(model.parameters()).device
except Exception:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEBUG: Model device:", device)
output = processor.set_text_prompt(state=inference_state, prompt=prompt)

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

def _debug_print_info(x, name: str):
	if x is None:
		print(f"DEBUG: {name} is None")
		return
	if hasattr(x, "shape"):
		try:
			print(f"DEBUG: {name} type={type(x)}, shape={x.shape}, dtype={getattr(x, 'dtype', None)}")
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