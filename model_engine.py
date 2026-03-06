import torch
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    CLIPProcessor,
    CLIPModel,
)
from PIL import Image
import io
import numpy as np
import cv2

# ── SegFormer (change detection) ─────────────────────────────────────────────
segformer_name = "nvidia/segformer-b0-finetuned-ade-512-512"
feature_extractor = SegformerImageProcessor.from_pretrained(segformer_name)
model = SegformerForSemanticSegmentation.from_pretrained(segformer_name)
model.eval()

# ── CLIP (satellite image validation) ────────────────────────────────────────
_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
_clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
_clip_model.eval()

# Prompts that CLIP scores the image against.
# The image is satellite if the mean score over satellite prompts
# exceeds the mean score over non-satellite prompts.
_SAT_PROMPTS = [
    "satellite imagery of earth surface captured from above",
    "aerial top-down view of terrain, land, or city",
    "remote sensing image showing forests, rivers, or urban areas",
    "overhead satellite view of desert, farmland, or coastline",
    "nadir view of earth showing roads, buildings, and vegetation",
]
_NON_SAT_PROMPTS = [
    "a photograph of a person or people",
    "a meme, funny image, or internet joke",
    "artwork, illustration, painting, or cartoon",
    "a document, screenshot, or page of text",
    "an indoor photo or close-up ground-level picture",
    "a selfie or portrait photo",
    "a food photo or product image",
]

def is_satellite_image(img_bytes):
    """
    Use CLIP zero-shot classification to decide whether an image is
    satellite / aerial / geographical imagery.

    Works by scoring the image against two sets of text prompts:
      - Satellite prompts  (top-down terrain, aerial, remote-sensing, …)
      - Non-satellite prompts (portraits, memes, art, documents, …)

    The image is accepted only when the average satellite-prompt score
    is higher than the average non-satellite-prompt score.

    Args:
        img_bytes: Raw image bytes to validate.

    Returns:
        bool: True if the image is likely satellite/geographical.
    """
    try:
        img_pil = Image.open(io.BytesIO(img_bytes))

        # TIFF is the standard remote-sensing container — always accept
        if img_pil.format == 'TIFF':
            return True

        img_rgb = img_pil.convert('RGB')
        all_prompts = _SAT_PROMPTS + _NON_SAT_PROMPTS

        inputs = _clip_processor(
            text=all_prompts,
            images=img_rgb,
            return_tensors='pt',
            padding=True,
        )

        with torch.no_grad():
            outputs = _clip_model(**inputs)

        # Shape: (1, num_prompts) — softmax over all prompts together
        probs = outputs.logits_per_image.softmax(dim=1)[0]

        n_sat = len(_SAT_PROMPTS)
        sat_score = probs[:n_sat].mean().item()
        non_sat_score = probs[n_sat:].mean().item()

        return sat_score > non_sat_score

    except Exception:
        return False

def generate_change_mask(before_img_bytes, after_img_bytes,
                         before_sar_bytes=None, after_sar_bytes=None):
    """
    Robust multi-signal change detection.

    Signals used (all normalised 0-1):
      1. LAB colour difference  – brightness-normalised via histogram matching
      2. Structural (Sobel)     – catches texture / edge changes
      3. SAR log-ratio          – cloud-penetrating, if SAR images supplied

    Final mask = weighted OR with morphological cleanup.
    """
    # ── Load images ───────────────────────────────────────────────────────────
    before_rgb = np.array(Image.open(io.BytesIO(before_img_bytes)).convert('RGB'))
    after_rgb  = np.array(Image.open(io.BytesIO(after_img_bytes)).convert('RGB'))

    # Resize to same shape
    if before_rgb.shape != after_rgb.shape:
        h = max(before_rgb.shape[0], after_rgb.shape[0])
        w = max(before_rgb.shape[1], after_rgb.shape[1])
        before_rgb = cv2.resize(before_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
        after_rgb  = cv2.resize(after_rgb,  (w, h), interpolation=cv2.INTER_LANCZOS4)

    H, W = before_rgb.shape[:2]

    # ── SIGNAL 1: Brightness-normalised LAB difference ────────────────────────
    def hist_match_L(src_lab, ref_lab):
        """Match the L channel of src to the distribution of ref."""
        s  = src_lab[:, :, 0].astype(np.float32)
        r  = ref_lab[:, :, 0].astype(np.float32)
        # std-based linear rescaling (fast, robust to outliers)
        mu_s, std_s = s.mean(), s.std() + 1e-6
        mu_r, std_r = r.mean(), r.std() + 1e-6
        matched = (s - mu_s) * (std_r / std_s) + mu_r
        out = src_lab.copy()
        out[:, :, 0] = np.clip(matched, 0, 255).astype(np.uint8)
        return out

    before_lab = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2LAB)
    after_lab  = cv2.cvtColor(after_rgb,  cv2.COLOR_RGB2LAB)

    # Match before's brightness distribution to after's so illumination changes cancel out
    before_lab_m = hist_match_L(before_lab, after_lab)

    # Per-channel difference in LAB (perceptually uniform)
    lab_diff = np.mean(np.abs(
        before_lab_m.astype(np.float32) - after_lab.astype(np.float32)
    ), axis=2)   # shape (H, W)

    # Normalise robustly using 98th percentile
    p98 = np.percentile(lab_diff, 98)
    lab_norm = np.clip(lab_diff / (p98 + 1e-6), 0, 1)

    # ── SIGNAL 2: Sobel structural difference ─────────────────────────────────
    gray_b = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_a = cv2.cvtColor(after_rgb,  cv2.COLOR_RGB2GRAY).astype(np.float32)

    def sobel_mag(img):
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    struct_b = sobel_mag(gray_b)
    struct_a = sobel_mag(gray_a)

    struct_diff = np.abs(struct_b - struct_a)
    p98s = np.percentile(struct_diff, 98)
    struct_norm = np.clip(struct_diff / (p98s + 1e-6), 0, 1)

    # ── SIGNAL 3: SAR log-ratio (optional) ───────────────────────────────────
    sar_norm = np.zeros((H, W), dtype=np.float32)
    sar_available = False

    if before_sar_bytes is not None and after_sar_bytes is not None:
        try:
            bsar = np.array(Image.open(io.BytesIO(before_sar_bytes)).convert('L')).astype(np.float32) + 1
            asar = np.array(Image.open(io.BytesIO(after_sar_bytes)).convert('L')).astype(np.float32) + 1
            if bsar.shape != (H, W):
                bsar = cv2.resize(bsar, (W, H))
                asar = cv2.resize(asar, (W, H))
            # Log-ratio: |log(after/before)|  — standard SAR change measure
            log_ratio = np.abs(np.log(asar / (bsar + 1e-6)))
            p98r = np.percentile(log_ratio, 98)
            sar_norm = np.clip(log_ratio / (p98r + 1e-6), 0, 1)
            sar_available = True
            print("✅ SAR signal active")
        except Exception as e:
            print(f"⚠️  SAR signal skipped: {e}")

    # ── FUSION ────────────────────────────────────────────────────────────────
    # Weighted combination: up-weight SAR when available (cuts through clouds)
    if sar_available:
        combined = 0.35 * lab_norm + 0.20 * struct_norm + 0.45 * sar_norm
    else:
        combined = 0.60 * lab_norm + 0.40 * struct_norm

    # ── Adaptive thresholding using Otsu on the combined map ──────────────────
    combined_u8 = (combined * 255).clip(0, 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(combined_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use a floor so quiet scenes don't threshold at 0
    threshold = max(float(otsu_val) / 255.0, 0.25)
    raw_mask = (combined > threshold).astype(np.uint8)

    # ── Morphological cleanup ─────────────────────────────────────────────────
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,  k_open)
    mask = cv2.morphologyEx(mask,     cv2.MORPH_CLOSE, k_close)

    # Remove tiny blobs (noise)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(20, int(H * W * 0.0001))  # 0.01% of image area
    clean = np.zeros_like(mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 1

    return Image.fromarray((clean * 255).astype(np.uint8), mode='L')


def overlay_mask(original_image, mask_image, color=(255, 0, 0), alpha=0.5):
    """
    Overlay the white regions of a binary mask onto the original image as a
    semi-transparent coloured highlight.

    Args:
        original_image: PIL Image – the 'After' image to draw on.
        mask_image:     PIL Image – B&W mask where white pixels mark changed areas.
        color:          RGB tuple for the highlight colour (default: red).
        alpha:          Opacity of the highlight layer, 0.0–1.0 (default: 0.5).

    Returns:
        PIL Image: Composited image with the highlight applied.
    """
    # --- normalise inputs to RGB numpy arrays ---
    original_rgb = np.array(original_image.convert('RGB'))
    mask_gray    = np.array(mask_image.convert('L'))

    # Resize mask to match the original image if dimensions differ
    if mask_gray.shape[:2] != original_rgb.shape[:2]:
        mask_gray = cv2.resize(
            mask_gray,
            (original_rgb.shape[1], original_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Binary mask: True where the mask is white (changed pixels)
    changed = mask_gray > 127                         # shape (H, W), bool

    # Build a solid colour layer the same size as the original
    colour_layer = np.zeros_like(original_rgb, dtype=np.uint8)
    colour_layer[changed] = color                     # apply colour only to changed pixels

    # Blend: result = original * (1 - alpha) + colour * alpha  — for changed pixels only
    composited = original_rgb.copy().astype(np.float32)
    composited[changed] = (
        original_rgb[changed].astype(np.float32) * (1.0 - alpha)
        + np.array(color, dtype=np.float32) * alpha
    )
    composited = np.clip(composited, 0, 255).astype(np.uint8)

    return Image.fromarray(composited, mode='RGB')
