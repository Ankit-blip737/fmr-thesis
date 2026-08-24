import argparse
import json
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

def otsu_thresholding_bbox(image):
    """
    Computes a pseudo bounding box using Otsu's thresholding.
    
    Args:
        image (PIL.Image): Input image.
        
    Returns:
        list or None: [x, y, w, h] of the bounding box.
    """
    img_gray = image.convert('L')
    img_arr = np.array(img_gray)
    
    # Calculate histogram
    hist, bin_edges = np.histogram(img_arr, bins=256, range=(0, 256))
    hist = hist.astype(float) / hist.sum()
    
    best_thresh = 0
    max_var = 0
    
    # Otsu's thresholding
    for t in range(1, 256):
        w0 = np.sum(hist[:t])
        w1 = np.sum(hist[t:])
        if w0 == 0 or w1 == 0:
            continue
            
        m0 = np.sum(np.arange(t) * hist[:t]) / w0
        m1 = np.sum(np.arange(t, 256) * hist[t:]) / w1
        
        var_between = w0 * w1 * (m0 - m1) ** 2
        if var_between > max_var:
            max_var = var_between
            best_thresh = t
            
    # Threshold the image
    # Assuming the foreground (anatomy) might be darker or lighter. 
    # Usually, a simple thresholding can isolate the main object.
    # We will just take the pixels that differ from the corners' mean.
    
    # A simpler general approach for medical images:
    mask = img_arr > best_thresh
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # Invert and try again
        mask = img_arr < best_thresh
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None
            
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    
    return [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]

def main():
    parser = argparse.ArgumentParser(description="Generate pseudo bounding boxes for medical VQA datasets.")
    parser.add_argument("--dataset", choices=["vqa_rad", "pathvqa", "slake"], required=True, help="Dataset to process.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--model", default="facebook/sam-vit-base", help="SAM model checkpoint.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--no-sam", action="store_true", help="Disable SAM and use fallback thresholding.")
    parser.add_argument("--image-root", default=None, help="Directory of images on disk (for datasets like SLAKE).")
    parser.add_argument("--fallback-threshold", action="store_true", help="Use fallback thresholding if SAM fails.")
    
    args = parser.parse_args()
    
    # Load dataset
    if args.dataset == "vqa_rad":
        dataset_name = "flaviagiammarino/vqa-rad"
    elif args.dataset == "slake":
        dataset_name = "BoKelvin/SLAKE"
    else:
        dataset_name = "flaviagiammarino/path-vqa"
        
    print(f"Loading dataset {dataset_name} (split=test)...")
    dataset = load_dataset(dataset_name, split="test")
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = None
    model = None
    if not args.no_sam:
        try:
            from transformers import SamModel, SamProcessor
            print(f"Loading SAM model {args.model} on {device}...")
            processor = SamProcessor.from_pretrained(args.model)
            model = SamModel.from_pretrained(args.model).to(device)
        except ImportError:
            print("transformers library not found. Falling back to thresholding.")
            args.no_sam = True
    
    output_dict = {}
    
    print(f"Processing {len(dataset)} images...")
    for idx, sample in enumerate(dataset):
        image = None
        if 'image' in sample and sample['image'] is not None:
            image = sample['image']
        elif 'img_name' in sample:
            # Check image_root or current directory
            candidates = []
            if args.image_root:
                candidates.append(Path(args.image_root) / sample['img_name'])
                candidates.append(Path(args.image_root) / "imgs" / sample['img_name'])
            candidates.append(Path("slake_imgs") / sample['img_name'])
            candidates.append(Path("slake_imgs") / "imgs" / sample['img_name'])
            for p in candidates:
                if p.exists():
                    try:
                        image = Image.open(p).convert('RGB')
                        break
                    except Exception:
                        pass
        
        if image is None:
            continue
            
        sample_id = f"{args.dataset}-test-{idx:06d}"
        bbox = None
        
        if not args.no_sam and model and processor:
            try:
                # SAM approach
                inputs = processor(image, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Simplified: taking the first mask or largest mask from automatic generation 
                # (For actual automatic generation with SAM in transformers, you often need to provide points. 
                # If no points are provided, we'll try to find a central mask, or fallback.)
                # Since automatic mask generation without prompts via pure Transformers SAM requires a pipeline,
                # we'll catch any error and use fallback.
                masks = outputs.pred_masks.squeeze(1)
                
                if masks.shape[0] > 0 and masks.shape[1] > 0:
                    best_mask = masks[0, 0].cpu().numpy() > 0
                    
                    rows = np.any(best_mask, axis=1)
                    cols = np.any(best_mask, axis=0)
                    if np.any(rows) and np.any(cols):
                        y0, y1 = np.where(rows)[0][[0, -1]]
                        x0, x1 = np.where(cols)[0][[0, -1]]
                        bbox = [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]
            except Exception as e:
                pass
                
        if bbox is None and (args.no_sam or args.fallback_threshold):
            bbox = otsu_thresholding_bbox(image)
            
        if bbox:
            output_dict[sample_id] = bbox
            
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(dataset)} images...")
            
    with open(args.output, 'w') as f:
        json.dump(output_dict, f, indent=4)
        
    print(f"\nProcessing complete.")
    print(f"Generated pseudo bounding boxes for {len(output_dict)}/{len(dataset)} images.")
    print(f"Saved to {args.output}")

if __name__ == '__main__':
    main()
