import argparse
import json
import os
from pathlib import Path
from PIL import Image
import numpy as np

def extract_bbox_from_mask(mask_path):
    """
    Extracts the bounding box from a binary mask image.
    
    Args:
        mask_path (str or Path): Path to the binary mask image.
        
    Returns:
        list or None: [x, y, w, h] of the bounding box, or None if no foreground is found.
    """
    try:
        mask_img = Image.open(mask_path).convert('L')
        mask = np.array(mask_img)
        
        # Threshold at 127
        mask = mask > 127
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
            
        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]
        
        return [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]
    except Exception as e:
        print(f"Error processing {mask_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract bounding boxes from SLAKE masks.")
    parser.add_argument("--mask-root", required=True, help="Path to SLAKE masks directory.")
    parser.add_argument("--image-root", help="Path to SLAKE imgs directory (optional).")
    parser.add_argument("--output", default="slake_bboxes.json", help="Output JSON path.")
    
    args = parser.parse_args()
    
    mask_root = Path(args.mask_root)
    output_dict = {}
    processed_count = 0
    
    print(f"Scanning masks in: {mask_root}")
    
    if not mask_root.exists():
        print(f"Error: Mask directory '{mask_root}' does not exist.")
        return

    # SLAKE masks are organized as: masks/<organ>/<image_name>.png
    for organ_dir in mask_root.iterdir():
        if not organ_dir.is_dir():
            continue
            
        organ_name = organ_dir.name
        
        for mask_file in organ_dir.rglob("*.png"):
            # Try to map back to SLAKE img_name if needed.
            # Usually SLAKE has images in e.g. xmlab123/source.jpg
            # Let's assume the mask file name is related to the image name.
            # Without specific structure, we'll use the file's stem or parent structure.
            # The instructions suggest: "The key should be the img_name field that SLAKE uses (e.g., xmlab123/source.jpg)."
            # Let's assume the mask relative path might be xmlab123/source.png, mapped from masks/<organ>/xmlab123/source.png
            # Or if it's just masks/<organ>/<image_name>.png, we use <image_name>.jpg
            
            img_name_key = mask_file.stem + ".jpg" # Default fallback
            # In SLAKE, sometimes the image is in a subfolder like xmlab1/source.jpg. 
            # If the mask is masks/Lung/xmlab1/source.png:
            try:
                rel_parts = mask_file.relative_to(organ_dir).parts
                img_name_key = "/".join(rel_parts).replace(".png", ".jpg")
            except ValueError:
                pass
                
            bbox = extract_bbox_from_mask(mask_file)
            if bbox:
                if img_name_key not in output_dict:
                    output_dict[img_name_key] = {}
                output_dict[img_name_key][organ_name] = bbox
                processed_count += 1
                
    with open(args.output, 'w') as f:
        json.dump(output_dict, f, indent=4)
        
    print(f"Extraction complete. Found bounding boxes for {processed_count} masks.")
    print(f"Saved to {args.output}")

if __name__ == '__main__':
    main()
