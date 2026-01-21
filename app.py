import os
import time
import warnings
import traceback
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision 
from torchvision import datasets, models, transforms
import timm 
from PIL import Image, ImageFilter
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, jaccard_score
)
from sklearn.model_selection import train_test_split
from torch.amp import autocast
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from statsmodels.stats.inter_rater import fleiss_kappa
import cv2
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import pearsonr
import gradio as gr
import tempfile
import copy

# Add model classes for safe loading
import torch.serialization
from torchvision.models.mobilenetv3 import MobileNetV3
from torchvision.models.shufflenetv2 import ShuffleNetV2
from timm.models.efficientnet import EfficientNet

print("--- Initializing Gradio App (18-Model Version) ---")

# 1. GLOBAL SETUP & CONFIGURATION
# Device Setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"--- Using device: {device} ---")

# Model & Data Config
resolutions = [256, 512, 1024]
model_families = ['mobilenet_v3_small', 'efficientnet_b0', 'shufflenet_v2_x1_0']
class_names = ['Benign', 'InSitu', 'Invasive', 'Normal']
num_classes = len(class_names)
input_size = 224


# File Paths
base_dir = os.getcwd()
models_fp32_dir = os.path.join(base_dir, "models", "fp32")
models_fp16_dir = os.path.join(base_dir, "models", "fp16")
processed_dir = os.path.join(base_dir, "data", "processed")
logs_dir = os.path.join(base_dir, "logs")
raw_test_dir = 'data/raw/bach/ICIAR2018_BACH_Challenge/Photos'
gt_csv_path = 'ground_truth_regions.csv'
patho_csv_path = 'pathologist_scores.csv'
heatmap_output_dir = os.path.join(logs_dir, "evaluation_heatmaps_gradio")

os.makedirs(logs_dir, exist_ok=True)
os.makedirs(heatmap_output_dir, exist_ok=True)
if not os.path.exists(models_fp32_dir): print(f"WARNING: FP32 models directory not found: {models_fp32_dir}")
if not os.path.exists(models_fp16_dir): print(f"WARNING: FP16 models directory not found: {models_fp16_dir}")
if not os.path.exists(processed_dir): print(f"WARNING: Processed data directory not found: {processed_dir}")


# Transforms
model_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. HELPER FUNCTIONS
def load_model_structure(model_family: str, num_classes: int = 4):
    """Loads the model *structure* (no weights) - CORRECTLY MATCHES TRAINING CELL 4"""
    model = None
    if model_family == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=None)
        in_features_final_layer = model.classifier[3].in_features # 1024
        original_first_linear = model.classifier[0] # This is Linear(in=576, out=1024)
        
        # Recreate the exact nn.Sequential from training
        model.classifier = nn.Sequential(
            original_first_linear,               # [0] Linear(576, 1024)
            nn.Hardswish(),                      # [1]
            # Set to 0.5 and inplace=False
            nn.Dropout(p=0.5, inplace=False),    
            nn.Linear(in_features_final_layer, num_classes) # [3] Linear(1024, 4)
        )
    elif model_family == 'efficientnet_b0':
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes) 
        in_features = model.get_classifier().in_features
        # This structure MUST match Cell 4
        model.classifier = nn.Sequential(
            # Set to 0.5 and inplace=False
            nn.Dropout(p=0.5, inplace=False), 
            nn.Linear(in_features, num_classes) # [1]
        )
    elif model_family == 'shufflenet_v2_x1_0':
        model = models.shufflenet_v2_x1_0(weights=None) 
        in_features = model.fc.in_features
        # This structure MUST match Cell 4
        model.fc = nn.Sequential(
            # Set to 0.5 (inplace was already False by default)
            nn.Dropout(p=0.5),  
            nn.Linear(in_features, num_classes) # [1]
        )
    else:
        raise ValueError(f"Unknown model family: {model_family}")
    return model

def get_target_layers(model, model_family):
    """Gets the correct target layer for Grad-CAM based on model family"""
    try:
        if model_family == 'mobilenet_v3_small':
            return [model.features[-1]]
        elif model_family == 'efficientnet_b0':
            # timm EfficientNet has a different structure
            return [model.conv_head]
        elif model_family == 'shufflenet_v2_x1_0':
            return [model.conv5]
    except Exception as e:
        print(f"Error getting target layer for {model_family}: {e}")
        # A generic fallback
        for layer in reversed(list(model.modules())):
            if isinstance(layer, torch.nn.Conv2d):
                return [layer]
        return None


def load_all_models():
    """Loads all 18 models into two global dictionaries."""
    
    torch.serialization.add_safe_globals([
        torchvision.models.mobilenetv3.MobileNetV3,
        timm.models.efficientnet.EfficientNet,
        torchvision.models.shufflenetv2.ShuffleNetV2
    ])
    
    g_models_fp32 = {}
    g_models_fp16 = {}
    
    for family in model_families:
        for res in resolutions:
            run_id = f"{family}_patch{res}"
            
            # Load FP32
            fp32_path = os.path.join(models_fp32_dir, f'model_{run_id}.pth')
            if os.path.exists(fp32_path):
                try:
                    model_fp32 = load_model_structure(family, num_classes)
                    checkpoint = torch.load(fp32_path, map_location=device)
                    model_fp32.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                    model_fp32.to(device).eval()
                    g_models_fp32[run_id] = model_fp32
                except Exception as e:
                    print(f"Error loading FP32 model {run_id}: {e}")
                    traceback.print_exc() 
            else:
                print(f"Warning: FP32 model not found: {fp32_path}")

            # Load FP16
            fp16_path = os.path.join(models_fp16_dir, f'model_{run_id}_fp16.pth')
            if os.path.exists(fp16_path):
                try:
                    model_fp16 = torch.load(fp16_path, map_location=device, weights_only=False)
                    model_fp16.to(device).eval()
                    g_models_fp16[run_id] = model_fp16
                except Exception as e:
                    print(f"Error loading FP16 model {run_id}: {e}")
                    traceback.print_exc() 
            else:
                print(f"Warning: FP16 model not found: {fp16_path}")
                
    print(f"--- Loaded {len(g_models_fp32)} FP32 and {len(g_models_fp16)} FP16 models ---")
    return g_models_fp32, g_models_fp16


def get_dataloaders(res: int, batch_size: int = 32, mode='eval'):
    data_path = os.path.join(processed_dir, f'patches_{res}')
    val_path = os.path.join(data_path, 'val')
    if not os.path.exists(val_path): print(f"Warning: Val path not found: {val_path}"); return None, None, None
    try:
        val_ds = datasets.ImageFolder(val_path, model_transform)
        if len(val_ds) == 0: print(f"Warning: Val dataset empty: {val_path}"); return None, None, None
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=min(4, os.cpu_count()), pin_memory=True)
        return None, val_loader, None
    except Exception as e: print(f"Error loading 'val' dataset at {data_path}: {e}"); return None, None, None


# compute_weights
# This function is updated to returns the metrics it calculates
def compute_weights(model_family, fp16=False, weighting_scheme='mrpe'):
    """Computes weights and ALSO returns the calculated val metrics."""
    global g_models_fp32, g_models_fp16
    
    scheme_name = weighting_scheme.upper()
    print(f"\n--- Computing {model_family} {'FP16' if fp16 else 'FP32'} Weights ({scheme_name}) ---")
    if weighting_scheme == 'equal': 
        return {res: 1.0 / len(resolutions) for res in resolutions}, {}, {}
    
    weights, accuracies, times_per_img = {}, {}, {}
    val_loaders = {}
    missing_loader = False
    
    for res in resolutions:
        _, val_loader, _ = get_dataloaders(res, mode='val')
        if val_loader is None: print(f"Error: Val loader missing for res {res}."); missing_loader=True; break
        val_loaders[res] = val_loader
        
    if missing_loader: 
        print("Fallback: Using equal weights.")
        return {r: 1.0 / len(resolutions) for r in resolutions}, {}, {}
    
    for res in resolutions:
        run_id = f"{model_family}_patch{res}"
        model_cache = g_models_fp16 if fp16 else g_models_fp32
        
        if run_id not in model_cache:
            print(f"  Error: Model {run_id} not in cache. Skipping weight calc for res {res}.")
            accuracies[res]=0.0; times_per_img[res]=float('inf')
            continue
            
        model = model_cache[run_id]
        model.eval()
        start_time = time.time(); preds, labels = [], []
        
        with torch.no_grad():
            for inputs, lbls in val_loaders[res]:
                inputs = inputs.to(device)
                if fp16 and device.type != 'cpu': inputs = inputs.half()
                
                with autocast(device_type=device.type, enabled=(fp16 and device.type != 'cpu')):
                    outputs = model(inputs)
                    
                preds.extend(torch.argmax(outputs, 1).cpu().numpy()); labels.extend(lbls.numpy())
                
        dataset_len = len(val_loaders[res].dataset)
        elapsed_time = time.time() - start_time
        time_per_img = elapsed_time / dataset_len if dataset_len > 0 else float('inf')
        acc = accuracy_score(labels, preds)
        accuracies[res] = acc; times_per_img[res] = time_per_img if time_per_img > 1e-9 else float('inf')
        print(f"  Res {res}: Acc={acc:.4f}, Time/Img={times_per_img[res]*1000:.2f} ms")
        
    raw_weights = {res: accuracies.get(res,0) / times_per_img.get(res,float('inf')) if weighting_scheme=='mrpe' else accuracies.get(res,0) for res in resolutions}
    total_weight = sum(w for w in raw_weights.values() if np.isfinite(w))
    
    if total_weight > 1e-9: weights = {res: (raw_weights[res] / total_weight) if np.isfinite(raw_weights.get(res,0)) else 0.0 for res in resolutions}
    else: print(f"Warn: Total weight {total_weight}. Using equal weights."); weights = {res: 1.0 / len(resolutions) for res in resolutions}
    
    print(f"Computed Weights ({model_family} {scheme_name}): {weights}")
    
    #Return calculated metrics
    return weights, accuracies, times_per_img


def extract_patches_temp(img: Image.Image, patch_size: int, stride: int):
    patches = []; img_width, img_height = img.size
    if img_width < patch_size or img_height < patch_size:
        new_size=(max(img_width, patch_size), max(img_height, patch_size))
        try: img=img.resize(new_size, Image.Resampling.LANCZOS); img_width, img_height=new_size
        except: print("Warn: Img resize failed."); return []
    for y in range(0, img_height-patch_size+1, stride):
        for x in range(0, img_width-patch_size+1, stride):
            try: patch=img.crop((x, y, x+patch_size, y+patch_size)); patches.append(patch)
            except: print("Warn: Patch crop failed."); continue
    if not patches:
        try: patch=img.resize((patch_size, patch_size), Image.Resampling.LANCZOS); patches.append(patch)
        except: print("Warn: Fallback resize failed."); return []
    return patches


def generate_heatmap(model, model_family, input_tensor_fp32, target_class):
    """
    Generates a Grad-CAM heatmap, robustly handling FP16/FP32 models
    by creating a guaranteed FP32 copy for CAM.
    """
    # model = model.to(device); # No longer needed, model is just for state_dict
    model.eval() # Ensure model is in eval mode
    
    try: 
        # 1. Create a new FP32 model structure
        model_fp32_for_cam = load_model_structure(model_family, num_classes)
        
        # 2. Load the state dict from the (potentially FP16) model
        # This upcasts the weights to FP32
        model_fp32_for_cam.load_state_dict(model.state_dict())
        model_fp32_for_cam.to(device).eval()
        
        # 3. Get target layers from the new FP32 model
        target_layers = get_target_layers(model_fp32_for_cam, model_family)
        if not target_layers: 
            print(f"ERROR: No target layers found for {model_family}"); 
            return None
    except Exception as e: 
        print(f"ERROR getting target layers or loading state_dict: {e}"); 
        return None

    # 4. Use a simple wrapper (no .float() needed inside, as model is already FP32)
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model): 
            super().__init__()
            self.model = model
        def forward(self, x): 
            return self.model(x) # Input tensor is already guaranteed FP32
    
    model_for_cam = ModelWrapper(model_fp32_for_cam)
    # No need for .float() or .to(device) again, already done
    
    cam = GradCAM(model=model_for_cam, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    
    try:
        # 5. Ensure input tensor is float() one last time
        grayscale_cam = cam(input_tensor=input_tensor_fp32.float().to(device), targets=targets)
        
        if grayscale_cam is None or len(grayscale_cam) == 0: 
            print("ERROR: GradCAM returned empty."); 
            return None
        
        cam_result = grayscale_cam[0, :]
        hm_normalized = cv2.normalize(cam_result, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return hm_normalized
    except Exception as e: 
        print(f"ERROR during GradCAM execution: {e}"); 
        traceback.print_exc(); 
        return None


def compute_iou(heatmap, gt_box_dict, threshold=0.5):
    if heatmap is None: return np.nan
    try:
        heatmap_bin = (heatmap > threshold).astype(np.uint8); h, w = heatmap.shape
        x_min=max(0,int(gt_box_dict['x_min']*w)); y_min=max(0,int(gt_box_dict['y_min']*h))
        x_max=min(w,int(gt_box_dict['x_max']*w)); y_max=min(h,int(gt_box_dict['y_max']*h))
        if y_min >= y_max or x_min >= x_max: return 0.0
        gt_mask = np.zeros_like(heatmap_bin); gt_mask[y_min:y_max, x_min:x_max] = 1
        intersection = np.logical_and(gt_mask, heatmap_bin).sum(); union = np.logical_or(gt_mask, heatmap_bin).sum()
        return intersection / union if union > 0 else 0.0
    except Exception as e: print(f"Error in compute_iou: {e}"); return np.nan


def compute_fleiss_kappa(df_ratings):
    try:
        n_categories = num_classes; n_subjects = len(df_ratings)
        count_matrix = np.zeros((n_subjects, n_categories), dtype=int)
        pathologist_cols = df_ratings.filter(regex='^p[0-9]+').columns
        if pathologist_cols.empty: return np.nan
        for i, (idx, row) in enumerate(df_ratings.iterrows()):
            ratings = pd.to_numeric(row[pathologist_cols].values, errors='coerce')
            for rating in ratings:
                if pd.notna(rating) and 0 <= rating < n_categories: count_matrix[i, int(rating)] += 1
        if np.any(count_matrix.sum(axis=1) == 0):
             valid_rows = count_matrix.sum(axis=1) > 0
             if not np.any(valid_rows): return np.nan
             count_matrix = count_matrix[valid_rows]
        if count_matrix.shape[0] < 2: return np.nan
        return fleiss_kappa(count_matrix, method='fleiss')
    except Exception as e: print(f"Error computing Fleiss' Kappa: {e}"); return np.nan


def add_gaussian_blur(image, sigma=2.0):
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def ensemble_inference(image_path, model_family, weights, fp16=False):
    global g_models_fp32, g_models_fp16
    img = Image.open(image_path).convert("RGB"); probs = []; device_to_use = device
    
    for res in resolutions:
        if weights.get(res, 0) == 0: 
            probs.append(None); continue
            
        try:
            run_id = f"{model_family}_patch{res}"
            model_cache = g_models_fp16 if fp16 else g_models_fp32
            
            if run_id not in model_cache:
                raise FileNotFoundError(f"Model {run_id} not found in global cache.")
                
            model = model_cache[run_id]
            model.eval()
            
            stride = res // 2; patches = extract_patches_temp(img, res, stride)
            if not patches: 
                probs.append(np.ones((1, num_classes))/num_classes); continue
                
            res_probs = []
            with torch.no_grad():
                for patch in patches:
                    input_tensor = model_transform(patch).unsqueeze(0).to(device_to_use)
                    if fp16 and device.type != 'cpu': input_tensor = input_tensor.half()
                    
                    with autocast(device_type=device.type, enabled=(fp16 and device.type != 'cpu')): 
                        output = model(input_tensor)
                        
                    res_probs.append(torch.softmax(output, 1).cpu().numpy())
                    
            if not res_probs: probs.append(np.ones((1, num_classes))/num_classes)
            else: avg_prob = np.mean(res_probs, axis=0); probs.append(avg_prob)
            
        except Exception as model_e: 
            print(f"ERROR inference {model_family} res {res}: {model_e}"); traceback.print_exc(); probs.append(None)
            
    valid_probs = [p for p in probs if p is not None and isinstance(p, np.ndarray) and p.shape == (1, num_classes)]
    if not valid_probs: return "Error", np.ones(num_classes)/num_classes
    
    fused_prob = np.zeros((1, num_classes)); valid_weights_sum = 0
    if weights is None: weights = {r: 1.0/len(resolutions) for r in resolutions}
    
    for i, res in enumerate(resolutions):
         if i < len(probs) and probs[i] is not None and isinstance(probs[i], np.ndarray) and probs[i].shape == (1, num_classes):
             current_weight = weights.get(res, 0);
             if current_weight > 0: fused_prob += probs[i] * current_weight; valid_weights_sum += current_weight
             
    if valid_weights_sum > 1e-6: fused_prob /= valid_weights_sum
    elif valid_probs: fused_prob = np.mean(valid_probs, axis=0) 
    else: fused_prob = np.ones((1, num_classes)) / num_classes
    
    pred_class_idx = np.argmax(fused_prob); pred_class = class_names[pred_class_idx]
    return pred_class, fused_prob.flatten()
# End Helper Functions


# 3. PRE-COMPUTE METRICS & CM
print("--- Loading all 18 models into memory ---")
g_models_fp32, g_models_fp16 = load_all_models()

#Load Metrics & CMs
print("--- Pre-loading all Metrics and Confusion Matrices ---")

g_eval_metrics = {'fp32': {}, 'fp16': {}, 'agreement_fp16': {}, 'kappa': np.nan}
g_cm_images = {}
placeholder_img_global = Image.new('RGB', (224, 224), color = 'grey')

try:
    # Load FP32 Test Metrics
    fp32_metrics_data = {
        'mobilenet_v3_small_patch256': {'Accuracy': 0.8118, 'F1-Macro': 0.8118, 'Precision-Macro': 0.8127, 'Recall-Macro': 0.8118},
        'mobilenet_v3_small_patch512': {'Accuracy': 0.8424, 'F1-Macro': 0.8422, 'Precision-Macro': 0.8437, 'Recall-Macro': 0.8424},
        'mobilenet_v3_small_patch1024': {'Accuracy': 0.8861, 'F1-Macro': 0.8867, 'Precision-Macro': 0.8885, 'Recall-Macro': 0.8861},
        'efficientnet_b0_patch256': {'Accuracy': 0.8546, 'F1-Macro': 0.8540, 'Precision-Macro': 0.8563, 'Recall-Macro': 0.8546},
        'efficientnet_b0_patch512': {'Accuracy': 0.8695, 'F1-Macro': 0.8690, 'Precision-Macro': 0.8702, 'Recall-Macro': 0.8695},
        'efficientnet_b0_patch1024': {'Accuracy': 0.8639, 'F1-Macro': 0.8648, 'Precision-Macro': 0.8669, 'Recall-Macro': 0.8639},
        'shufflenet_v2_x1_0_patch256': {'Accuracy': 0.7968, 'F1-Macro': 0.7965, 'Precision-Macro': 0.7971, 'Recall-Macro': 0.7968},
        'shufflenet_v2_x1_0_patch512': {'Accuracy': 0.8410, 'F1-Macro': 0.8406, 'Precision-Macro': 0.8420, 'Recall-Macro': 0.8410},
        'shufflenet_v2_x1_0_patch1024': {'Accuracy': 0.7806, 'F1-Macro': 0.7823, 'Precision-Macro': 0.8008, 'Recall-Macro': 0.7806},
    }
    g_eval_metrics['fp32'] = fp32_metrics_data

    # Load FP16 Test Metrics
    fp16_metrics_data = {
        'mobilenet_v3_small_patch256': {'Accuracy': 0.9000, 'F1-Macro': 0.8994, 'Lat': 79.65},
        'mobilenet_v3_small_patch512': {'Accuracy': 0.9167, 'F1-Macro': 0.9160, 'Lat': 80.65},
        'mobilenet_v3_small_patch1024': {'Accuracy': 0.9000, 'F1-Macro': 0.8994, 'Lat': 83.27},
        'efficientnet_b0_patch256': {'Accuracy': 0.9500, 'F1-Macro': 0.9499, 'Lat': 65.91},
        'efficientnet_b0_patch512': {'Accuracy': 0.9000, 'F1-Macro': 0.8999, 'Lat': 65.59},
        'efficientnet_b0_patch1024': {'Accuracy': 0.9167, 'F1-Macro': 0.9160, 'Lat': 66.96},
        'shufflenet_v2_x1_0_patch256': {'Accuracy': 0.8833, 'F1-Macro': 0.8841, 'Lat': 10.88},
        'shufflenet_v2_x1_0_patch512': {'Accuracy': 0.9167, 'F1-Macro': 0.9161, 'Lat': 11.08},
        'shufflenet_v2_x1_0_patch1024': {'Accuracy': 0.7500, 'F1-Macro': 0.7514, 'Lat': 12.40},
    }
    g_eval_metrics['fp16'] = fp16_metrics_data
    
    #Load Agreement Metrics
    g_eval_metrics['agreement_fp16'] = {
        'mobilenet_v3_small': 0.8500,
        'efficientnet_b0': 0.8500,
        'shufflenet_v2_x1_0': 0.8167,
    }
    
    # Load Fleiss Kappa
    if os.path.exists(patho_csv_path):
        patho_df_kappa = pd.read_csv(patho_csv_path)
        g_eval_metrics['kappa'] = compute_fleiss_kappa(patho_df_kappa.filter(regex='^p[0-9]+'))
    else:
        print(f"Warning: {patho_csv_path} not found for Kappa.")

    # Load all 18 Confusion Matrices
    print("Loading Confusion Matrices...")
    for family in model_families:
        for res in resolutions:
            for precision in ['fp32', 'fp16']:
                key = f'{family}_patch{res}_{precision}'
                cm_path = os.path.join(logs_dir, f'cm_{family}_patch{res}_{precision}.png')
                if os.path.exists(cm_path):
                    g_cm_images[key] = Image.open(cm_path)
                else:
                    g_cm_images[key] = None # Will use placeholder if not found
    print(f"Loaded {len([img for img in g_cm_images.values() if img is not None])} CM images.")

except Exception as metric_e: 
    print(f"Error pre-loading metrics: {metric_e}")
    traceback.print_exc()


print("--- Computing initial weights for all model families ---")
weights_fp32_global = {}
weights_fp16_global = {}
for family in model_families:
    # Capture metrics from compute_weights
    weights_fp32, acc_fp32, lat_fp32 = compute_weights(family, fp16=False, weighting_scheme='mrpe')
    weights_fp16, acc_fp16, lat_fp16 = compute_weights(family, fp16=True, weighting_scheme='mrpe')
    
    weights_fp32_global[family] = weights_fp32
    weights_fp16_global[family] = weights_fp16
    
    # Store the FP32 validation latency 
    for res in resolutions:
        key = f"{family}_patch{res}"
        if key in g_eval_metrics['fp32'] and lat_fp32.get(res):
            g_eval_metrics['fp32'][key]['Lat'] = lat_fp32[res] * 1000 # Convert to ms
        # We can also add the validation accuracy if we want
        # g_eval_metrics['fp32'][key]['Val_Acc'] = acc_fp32.get(res)
        # g_eval_metrics['fp16'][key]['Val_Acc'] = acc_fp16.get(res)

print("--- Initial Weights Computed & Metrics Captured ---")


# 4. GRADIO classify_image FUNCTION
# Updated to return real metrics and 3 CMs
def classify_image(image, model_family, precision_str, add_noise, sigma, ablate_res_str):
    global weights_fp32_global, weights_fp16_global
    global g_eval_metrics, g_cm_images, placeholder_img_global
    global g_models_fp32, g_models_fp16
    
    pred_class_output = "Error"; probs_img = None; fused_vis = None
    hm_256_vis, hm_512_vis, hm_1024_vis = None, None, None
    metrics_output = "Metrics calculation failed."
    cm_256_vis, cm_512_vis, cm_1024_vis = None, None, None
    temp_path = None
    
    placeholder_img = placeholder_img_global # Use the global placeholder
    
    use_fp16 = (precision_str == "FP16")
    model_cache = g_models_fp16 if use_fp16 else g_models_fp32
    precision_key = precision_str.lower()

    try:
        if image is None: raise ValueError("No image provided.")
        image_to_process = image.copy()
        if add_noise:
            print(f"Applying Gaussian blur with sigma={sigma}")
            image_to_process = add_gaussian_blur(image_to_process, sigma=sigma)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_path = tmp.name; image_to_process.save(temp_path, format='PNG')
            print(f"Temp image saved: {temp_path}")

        weights_base = weights_fp16_global[model_family] if use_fp16 else weights_fp32_global[model_family]
        weights_to_use = weights_base.copy()
        
        if ablate_res_str != "None":
            try:
                ablate_res = int(ablate_res_str)
                print(f"Ablating resolution: {ablate_res} for {model_family}")
                weights_to_use[ablate_res] = 0.0 # Set ablated weight to zero
                
                current_sum = sum(w for r, w in weights_to_use.items() if r in resolutions)
                if current_sum < 1e-9:
                     print("Warn: Ablation leaves zero weight. Using equal for remaining.")
                     active_res = [r for r in resolutions if r != ablate_res]
                     weights_to_use = {res: (1.0/len(active_res) if res != ablate_res else 0.0) for res in resolutions} if active_res else {}
                else: 
                     weights_to_use = {res: (w / current_sum) for res, w in weights_to_use.items()}
            except ValueError: print(f"Error: Invalid ablation resolution '{ablate_res_str}'. Using original.")

        print(f"Using Weights for {model_family} ({precision_str}): {weights_to_use}")
        
        pred_class, probs_flat = ensemble_inference(temp_path, model_family, weights=weights_to_use, fp16=use_fp16)
        pred_class_output = pred_class
        pred_idx = class_names.index(pred_class) if pred_class in class_names else -1

        if probs_flat is not None and len(probs_flat) == num_classes:
            with plt.style.context('seaborn-v0_8-darkgrid'):
                probs_fig, ax = plt.subplots(figsize=(6, 4))
                bars = sns.barplot(x=class_names, y=probs_flat, ax=ax, palette="viridis")
                ax.set_title('Class Probabilities')
                ax.set_ylabel('Probability')
                ax.set_ylim(0, 1.05)
                for bar in bars.patches:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2., bar.get_height(),
                        f'{bar.get_height():.2f}', ha='center', va='bottom',
                        fontsize=10, color='black'
                    )
            probs_buf = io.BytesIO()
            probs_fig.savefig(probs_buf, format='png', bbox_inches='tight', facecolor=probs_fig.get_facecolor())
            plt.close(probs_fig); probs_buf.seek(0)
            probs_img = Image.open(probs_buf)
        else:
             print("Warning: Invalid probabilities array received.")

        print("Generating heatmaps...")
        input_tensor_heatmap_fp32 = model_transform(image_to_process).unsqueeze(0).to(device).float()
        rgb_img_heatmap = np.array(image_to_process.resize((input_size, input_size))) / 255.0
        heatmaps_norm = []; heatmap_images = {}

        for res in resolutions:
            if weights_to_use.get(res, 0) == 0:
                 print(f"  Skipping heatmap res {res} (ablated/zero weight)."); heatmap_images[res]=None; continue
            
            run_id = f"{model_family}_patch{res}"
            if run_id not in model_cache:
                print(f"  Skipping heatmap res {res}: Model {run_id} not in cache."); heatmap_images[res]=None; continue
                
            model_hm = model_cache[run_id]
            
            if pred_idx != -1:
                # Use the new robust generate_heatmap function
                hm_norm = generate_heatmap(model_hm, model_family, input_tensor_heatmap_fp32, pred_idx)
                if hm_norm is not None:
                    heatmaps_norm.append(hm_norm); vis = show_cam_on_image(rgb_img_heatmap, hm_norm, use_rgb=True, image_weight=0.5)
                    heatmap_images[res] = Image.fromarray(vis)
                else: 
                    print(f"  Heatmap generation failed for res {res}.")
                    heatmap_images[res] = None
            else: 
                print(f"  Skipping heatmap res {res}: Invalid prediction index."); 
                heatmap_images[res]=None

        if heatmaps_norm:
            fused_hm_weighted = np.zeros_like(heatmaps_norm[0]); total_hm_weight = 0; temp_hm_idx = 0
            
            # Match weights to the heatmaps that were actually generated
            active_res_with_hm = [res for res in resolutions if heatmap_images.get(res) is not None]
            active_weights = [weights_to_use.get(res, 0) for res in active_res_with_hm]
            total_hm_weight = sum(active_weights)

            if total_hm_weight > 1e-6:
                for i, w in enumerate(active_weights):
                    fused_hm_weighted += heatmaps_norm[i] * (w / total_hm_weight)
            elif heatmaps_norm: # Fallback to equal
                fused_hm_weighted = np.mean(heatmaps_norm, axis=0) 
            else:
                fused_hm_weighted = np.zeros_like(heatmaps_norm[0])
                
            fused_hm_final = cv2.normalize(fused_hm_weighted, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            fused_vis_arr = show_cam_on_image(rgb_img_heatmap, fused_hm_final, use_rgb=True, image_weight=0.5)
            fused_vis = Image.fromarray(fused_vis_arr)
            proxy_gt_box = {'x_min': 0.3, 'y_min': 0.3, 'x_max': 0.7, 'y_max': 0.7}
            iou_val = compute_iou(fused_hm_final, proxy_gt_box)
            iou_str = f"{iou_val:.4f}" if pd.notna(iou_val) else "N/A"
        else: 
            print("No heatmaps were generated, fused heatmap cannot be created.")
            fused_vis = None; iou_str = "N/A"

        # Dynamically build metrics string
        metrics_256 = g_eval_metrics.get(precision_key, {}).get(f'{model_family}_patch256', {})
        metrics_512 = g_eval_metrics.get(precision_key, {}).get(f'{model_family}_patch512', {})
        metrics_1024 = g_eval_metrics.get(precision_key, {}).get(f'{model_family}_patch1024', {})
        
        acc_256 = metrics_256.get('Accuracy', 'N/A')
        f1_256 = metrics_256.get('F1-Macro', 'N/A')
        lat_256 = metrics_256.get('Lat', 'N/A')

        acc_512 = metrics_512.get('Accuracy', 'N/A')
        f1_512 = metrics_512.get('F1-Macro', 'N/A')
        lat_512 = metrics_512.get('Lat', 'N/A')
        
        acc_1024 = metrics_1024.get('Accuracy', 'N/A')
        f1_1024 = metrics_1024.get('F1-Macro', 'N/A')
        lat_1024 = metrics_1024.get('Lat', 'N/A')

        agreement = g_eval_metrics.get('agreement_fp16', {}).get(model_family, 'N/A')
        kappa = g_eval_metrics.get('kappa', 'N/A')
        
        # Helper to format metrics
        def f_val(val, fmt): 
            return f"{val:{fmt}}" if isinstance(val, (int, float)) else str(val)

        metrics_output = (
            f"--- Model: {model_family} ({precision_str}) ---\n\n"
            f"--- Patch 256 ---\n"
            f"  Accuracy (Test): {f_val(acc_256, '.4f')}\n"
            f"  F1-Macro (Test): {f_val(f1_256, '.4f')}\n"
            f"  Latency (Eval):  {f_val(lat_256, '.2f')} ms\n\n"
            f"--- Patch 512 ---\n"
            f"  Accuracy (Test): {f_val(acc_512, '.4f')}\n"
            f"  F1-Macro (Test): {f_val(f1_512, '.4f')}\n"
            f"  Latency (Eval):  {f_val(lat_512, '.2f')} ms\n\n"
            f"--- Patch 1024 ---\n"
            f"  Accuracy (Test): {f_val(acc_1024, '.4f')}\n"
            f"  F1-Macro (Test): {f_val(f1_1024, '.4f')}\n"
            f"  Latency (Eval):  {f_val(lat_1024, '.2f')} ms\n\n"
            f"--- Other Metrics ---\n"
            f"  Model-Pathologist Agreement (FP16 Ensemble): {f_val(agreement, '.4f')}\n"
            f"  Inter-Pathologist Kappa (Eval): {f_val(kappa, '.4f')}\n"
            f"  Fused Heatmap IoU (Proxy GT): {iou_str}"
        )
        
        # Load the 3 Confusion Matrices
        cm_256_vis = g_cm_images.get(f'{model_family}_patch256_{precision_key}', placeholder_img)
        cm_512_vis = g_cm_images.get(f'{model_family}_patch512_{precision_key}', placeholder_img)
        cm_1024_vis = g_cm_images.get(f'{model_family}_patch1024_{precision_key}', placeholder_img)

        hm_256_vis = heatmap_images.get(256); hm_512_vis = heatmap_images.get(512); hm_1024_vis = heatmap_images.get(1024)

    except Exception as e: 
        print(f"!!! Error in classify_image: {e}"); traceback.print_exc()
        pred_class_output = f"Error: {e}"
        metrics_output = f"Error generating metrics: {e}"
    finally:
        if temp_path and os.path.exists(temp_path):
            try: os.remove(temp_path); print(f"Temp image cleaned: {temp_path}")
            except: pass

    return (str(pred_class_output), probs_img or placeholder_img, fused_vis or placeholder_img,
            hm_256_vis or placeholder_img, hm_512_vis or placeholder_img, hm_1024_vis or placeholder_img,
            str(metrics_output), 
            cm_256_vis or placeholder_img, cm_512_vis or placeholder_img, cm_1024_vis or placeholder_img)


# 5. GRADIO INTERFACE DEFINITION
with gr.Blocks(theme=gr.themes.Soft()) as iface:

    gr.Markdown("# BACH Classifier for Clinics (MRPE)")

    with gr.Row():

        # Column 1: Input Image & Options
        with gr.Column(scale=1): 
            
            image_input = gr.Image(
                type="pil", label="Image Preview",
                height=400, interactive=False
            )
            
            upload_btn = gr.UploadButton(
                "📁 Upload Histology Image (.tif, .png, .jpg)",
                file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"], 
                file_count="single"
            )

            remove_btn = gr.Button("❌ Remove Image")

            file_info = gr.Textbox(
                label="File Info", interactive=False, visible=False
            )

            def handle_upload(file):
                if file is None: return None, gr.update(visible=False)
                try:
                    pil_img = Image.open(file.name)
                    if pil_img.mode != "RGB": preview_img = pil_img.convert("RGB")
                    else: preview_img = pil_img
                    info_text = f"📄 {os.path.basename(file.name)} | 📏 {pil_img.size[0]}x{pil_img.size[1]} | 🎨 {pil_img.mode}"
                    return preview_img, gr.update(value=info_text, visible=True)
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    return None, gr.update(value=error_msg, visible=True)

            def handle_remove():
                return None, gr.update(value="", visible=False)

            upload_btn.upload(
                fn=handle_upload, inputs=upload_btn, outputs=[image_input, file_info], show_progress="full"
            )

            remove_btn.click(
                fn=handle_remove, inputs=[], outputs=[image_input, file_info]
            )
            
            precision_input = gr.Radio(
                ["FP32", "FP16"],
                label="Select Precision",
                value="FP16" if device.type != 'cpu' else 'FP32'
            )
            
            model_family_input = gr.Dropdown(
                model_families,
                label="Select Model Architecture",
                value=model_families[1] # Default to EfficientNet
            )

            with gr.Accordion("Advanced Settings", open=False):
                 add_noise = gr.Checkbox(label="Add Gaussian Noise (Robustness Test)", value=False)
                 sigma = gr.Slider(minimum=0.5, maximum=5.0, step=0.1, value=2.0, label="Noise Sigma", visible=False)
                 ablate_res = gr.Dropdown(choices=["None"] + [str(r) for r in resolutions], label="Ablate Resolution (Contribution Test)", value="None")

            add_noise.change(fn=lambda x: gr.update(visible=x), inputs=add_noise, outputs=sigma)

            submit_btn = gr.Button("Classify Image", variant="primary")

        # Column 2: Results Tabs
        with gr.Column(scale=2): 
            with gr.Tabs():
                with gr.Tab("📊 Prediction & Probabilities"):
                    pred_class_out = gr.Textbox(label="Predicted Class")
                    probs_img_out = gr.Image(label="Class Probabilities", interactive=False)

                with gr.Tab("🔥 Heatmaps (Explainability)"):
                    fused_heatmap_out = gr.Image(label="Fused Ensemble Heatmap", interactive=False)
                    with gr.Row():
                        hm_256_out = gr.Image(label="Heatmap (Res 256)", interactive=False)
                        hm_512_out = gr.Image(label="Heatmap (Res 512)", interactive=False)
                        hm_1024_out = gr.Image(label="Heatmap (Res 1024)", interactive=False)

                with gr.Tab("📈 Metrics & Validation"):
                     metrics_text_out = gr.Textbox(
                         label="Performance & Validation Metrics (Pre-computed)", 
                         lines=13, # Increased lines
                         interactive=False
                     )
                     with gr.Row(): # New row for the 3 CMs
                        cm_256_out = gr.Image(label="CM (Res 256)", interactive=False)
                        cm_512_out = gr.Image(label="CM (Res 512)", interactive=False)
                        cm_1024_out = gr.Image(label="CM (Res 1024)", interactive=False)
    
    # Update click handler outputs
    submit_btn.click(
        fn=classify_image,
        inputs=[
            image_input,
            model_family_input,
            precision_input,
            add_noise,
            sigma,
            ablate_res
        ],
        outputs=[
            pred_class_out, probs_img_out, fused_heatmap_out,
            hm_256_out, hm_512_out, hm_1024_out,
            metrics_text_out, 
            cm_256_out, cm_512_out, cm_1024_out # Updated outputs
        ],
        show_progress="full"
    )


# 6. GRADIO LAUNCH
if __name__ == "__main__":
    print("\n--- NOTE FOR GRADIO ---")
    print("If Gradio fails or shows errors, check package versions (gradio, gradio_client).")
    print("Try: pip install --upgrade gradio gradio_client\n")
    print("--- Starting Gradio Interface ---")
    try: iface.launch(server_port=7863, share=True, inbrowser=True)
    except OSError as e:
        print(f"Port 7863 likely in use: {e}. Trying 7864.")
        try: iface.launch(server_port=7864, share=True, inbrowser=True)
        except Exception as e2: print(f"FATAL: Failed Gradio launch on both ports: {e2}")
    except Exception as e_launch: print(f"FATAL: Unexpected Gradio launch error: {e_launch}")