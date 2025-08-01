# Import modules
import os
import random
import torch
import pandas as pd
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image, UnidentifiedImageError
import warnings


# Initialize SAM2
checkpoint = "/home/ubuntu/Udi_SAM2/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device="cpu"))

# Configuration
consolidated_csv = 'consolidated_coordinates.csv'  # Changed to single CSV file
image_dirs = {'blink': 'Blink_tester', 'open': 'Open_tester'}
output_csv = "eye_segmentation_results_contour_analysis_complete_sam2.csv"
error_csv = "processing_errors_with_sam2.csv"

# Data storage
results = []
error_log = []

# Updated function. Now look at cases where masks > 2 as well
def has_two_non_overlapping_regions(mask, min_size_ratio=0.3): # Was 0.5 earlier, now modified to 0.3
    """
    Check if mask has at least two non-overlapping regions where the largest two
    are roughly the same size.
    
    Args:
        mask (ndarray): Binary mask image
        min_size_ratio (float): Minimum ratio between smaller and larger region (default: 0.5)
        
    Returns:
        bool: True if the two largest regions meet the size ratio criteria
    """
    labeled_mask, num_features = ndimage.label(mask)
    
    # Need at least 2 regions to compare
    if num_features < 2:
        return False
    
    # Calculate sizes of all regions
    sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
    
    # Get indices of the two largest regions
    largest_two_indices = np.argsort(sizes)[-2:]
    size1, size2 = sizes[largest_two_indices[0]], sizes[largest_two_indices[1]]
    
    # Calculate size ratio (smaller/larger)
    size_ratio = min(size1, size2) / max(size1, size2)
    
    return size_ratio >= min_size_ratio


def calculate_eccentricity(mask):
    """Calculate eccentricity of the second largest non-overlapping region.
       Returns 1 if region is not valid or fitting fails."""
    labeled_mask, num_features = ndimage.label(mask)

    if num_features < 2:
        return 1

    # Get second largest region
    region_areas = [(i, np.sum(labeled_mask == i)) for i in range(1, num_features + 1)]
    region_areas.sort(key=lambda x: x[1], reverse=True)
    second_largest_label = region_areas[1][0]

    region_mask = (labeled_mask == second_largest_label).astype(np.uint8)
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or len(contours[0]) < 5:
        return 1

    try:
        ellipse = cv2.fitEllipse(contours[0])
        (_, (axis1, axis2), _) = ellipse
        major_axis = max(axis1, axis2)
        minor_axis = min(axis1, axis2)

        if major_axis == 0 or minor_axis == 0:
            return 1

        ratio = minor_axis / major_axis
        ratio = np.clip(ratio, 0.0, 1.0)  # Prevent sqrt of negative
        eccentricity = np.sqrt(1 - ratio ** 2)
        return eccentricity

    except cv2.error:
        return 1




def select_best_mask(masks, scores):
    """Select best mask according to criteria, including eccentricity condition.
       Returns tuple of (mask, score, eccentricity)"""
    top_indices = np.argsort(scores)[-3:][::-1]  # Top 3 masks by score
    top_masks = [masks[i] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]

    for mask, score in zip(top_masks, top_scores):
        mask_area = np.sum(mask)
        if mask_area < 75000 and score > 0.02 and has_two_non_overlapping_regions(mask):
            ecc = calculate_eccentricity(mask)
            if ecc < 0.9:
                return mask, score, ecc  # Return eccentricity with override

    # Fallback: return best scoring mask with its eccentricity
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    best_ecc = calculate_eccentricity(best_mask)
    return best_mask, scores[best_idx], best_ecc


# # Use if needed
# def calculate_region_aspect_ratio(mask, region_label):
#     """Calculate aspect ratio (height/width) of a specific labeled region."""
#     region_mask = (mask == region_label)
#     contours, _ = cv2.findContours(region_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return 0
    
#     cnt = max(contours, key=cv2.contourArea)
#     _, (width, height), _ = cv2.minAreaRect(cnt)
#     return height / (width + 1e-6)  # height/width ratio


# # Use as fallback
# def select_best_mask(masks, scores):
#     """Select best mask according to new criteria with additional quality checks."""
#     top_indices = np.argsort(scores)[-3:][::-1]  # Get top 3 masks by score
#     top_masks = [masks[i] for i in top_indices]
#     top_scores = [scores[i] for i in top_indices]
    
#     # First check: look for masks meeting all criteria
#     for i, (mask, score) in enumerate(zip(top_masks, top_scores)):
#         mask_area = np.sum(mask)
#         if (mask_area < 75000 and 
#             has_two_non_overlapping_regions(mask) and 
#                                            # New condition can be added regarding eccentricity
#             score >= 0.02):
#             return mask, score
    
#     # Second check: if no masks meet all criteria, return the highest scoring mask
#     best_idx = np.argmax(scores)
#     return masks[best_idx], scores[best_idx]

    


def log_error(subfolder, img_name, error_type, details=""):
    """Record processing errors to error_log."""
    error_log.append({
        'subfolder': subfolder,
        'image_name': img_name,
        'error_type': error_type,
        'details': str(details)[:200]  # Truncate long error messages
    })
    print(f"! ERROR [{error_type}] {subfolder}/{img_name}: {details}")

def process_images_with_sam2(label, num_images=None):
    """Process images with comprehensive error handling."""
    collected = 0
    
    # Load consolidated CSV once at the beginning
    try:
        coord_df = pd.read_csv(consolidated_csv)
    except Exception as e:
        log_error("GLOBAL", "ALL", "CONSOLIDATED_CSV_LOAD_ERROR", e)
        return

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    for subfolder in sorted(os.listdir(image_dirs[label])):
        subfolder_path = os.path.join(image_dirs[label], subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        images = os.listdir(subfolder_path)
        for img_name in images:
            if num_images is not None and collected >= num_images:
                return

            try:
                # Validate CSV entry - now looking in consolidated file
                row = coord_df[coord_df['actual_filename'] == img_name]  # Changed column name
                if row.empty:
                    log_error(subfolder, img_name, "MISSING_CSV_ENTRY")
                    continue

                row = row.iloc[0]
                try:
                    lx, ly = int(row['abs_lx']), int(row['abs_ly'])
                    rx, ry = int(row['abs_rx']), int(row['abs_ry'])
                except (ValueError, KeyError) as e:
                    log_error(subfolder, img_name, "INVALID_COORDINATES", e)
                    continue

                # Check file extension before loading
                img_ext = os.path.splitext(img_name)[-1].lower()
                if img_ext not in valid_extensions:
                    log_error(subfolder, img_name, "INVALID_FILE_EXTENSION", f"Extension {img_ext} not supported")
                    continue

                # Load and validate image
                img_path = os.path.join(subfolder_path, img_name)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        image = cv2.imread(img_path)
                        if image is None:
                            raise AttributeError("cv2.imread returned None (possibly unreadable/corrupt)")
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except (UnidentifiedImageError, Warning, cv2.error, TypeError, OSError, AttributeError) as e:
                    log_error(subfolder, img_name, "IMAGE_LOAD_ERROR", e)
                    continue

                # Process with SAM2
                predictor.set_image(image_rgb)
                masks, scores, logits = predictor.predict(
                    point_coords=np.array([[lx, ly], [rx, ry]]),
                    point_labels=np.array([1, 1]),
                    multimask_output=True
                )

                best_mask, best_score, eccentricity = select_best_mask(masks, scores)
                results.append({
                    'image_name': img_name,
                    'label': label,
                    'lx': lx, 'ly': ly,
                    'rx': rx, 'ry': ry,
                    'mask_area': int(np.sum(best_mask)),
                    'mask_score': float(best_score),
                    'eccentricity': float(eccentricity),  # New field
                    'subfolder': subfolder
                })

                collected += 1
                print(f"[{label.upper()}] {subfolder}/{img_name} - Processed")

            except Exception as e:
                log_error(subfolder, img_name, "PROCESSING_ERROR", e)
                continue


# Main execution
if __name__ == "__main__":
    print("Starting processing...")
    process_images_with_sam2('blink',5)
    process_images_with_sam2('open',5)

    # Save successful results
    if results:
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"Saved {len(results)} successful results to {output_csv}")
    else:
        print("No images processed successfully")

    # Save error log
    if error_log:
        pd.DataFrame(error_log).to_csv(error_csv, index=False)
        print(f"Saved {len(error_log)} errors to {error_csv}")
    else:
        print("No errors encountered")

    print("Processing complete")