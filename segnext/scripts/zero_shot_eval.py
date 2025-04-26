import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import gc
import time
import pickle
import sys
from pathlib import Path
import torch
from easydict import EasyDict as edict
import yaml
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

segnext_path = Path(__file__).parent.parent.as_posix()
sys.path.insert(0, segnext_path)

from isegm.data.my_datasets import DavisDataset, HQSeg44kDataset, HypersimDataset
from isegm.model.is_plainvit_model import PlainVitModel
from isegm.inference.predictor import BasePredictor
from isegm.inference.utils import load_is_model, get_iou
from isegm.inference.clicker import RandomClicker as Clicker, Click


def iou_zero_loops(
    masks1: torch.Tensor, masks2: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Zero for‑loops. masks1: (M,H,W), masks2: (N,H,W).
    Returns IoU: (M,N).
    """
    M, H, W = masks1.shape
    N = masks2.shape[0]
    # reshape for broadcasting
    m1 = masks1.view(M, 1, H, W).float()
    m2 = masks2.view(1, N, H, W).float()
    # intersection & union
    inter = (m1 * m2).sum(dim=(2, 3))  # (M,N)
    area1 = m1.sum(dim=(2, 3))  # (M,1)
    area2 = m2.sum(dim=(2, 3))  # (1,N)
    union = area1 + area2 - inter  # (M,N)
    return inter / (union + eps)


def iou_one_loop(
    masks1: torch.Tensor, masks2: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    One for‑loop over masks1. Vectorized over masks2.
    """
    M, H, W = masks1.shape
    N = masks2.shape[0]
    iou = torch.zeros((M, N), dtype=torch.float, device=masks1.device)
    area2 = masks2.view(N, -1).sum(dim=1)  # (N,)
    for i in range(M):
        m = masks1[i].float().view(1, H, W)  # (1,H,W)
        inter = (m * masks2).view(N, -1).sum(dim=1)  # (N,)
        area1 = m.sum()
        union = area1 + area2 - inter  # (N,)
        iou[i] = inter / (union + eps)
    return iou


def iou_two_loops(
    masks1: torch.Tensor, masks2: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Two for‑loops over masks1 and masks2. Lowest memory overhead.
    """
    M, H, W = masks1.shape
    N = masks2.shape[0]
    iou = torch.zeros((M, N), dtype=torch.float, device=masks1.device)
    for i in range(M):
        m = masks1[i].float()
        area1 = m.sum()
        for j in range(N):
            p = masks2[j].float()
            inter = (m * p).sum()
            area2 = p.sum()
            union = area1 + area2 - inter
            iou[i, j] = inter / (union + eps)
    return iou


def incremental_mIoU(M, P_o, IoU):
    """
    M      : list of ground‐truth masks, |M| = M_n
    P_o    : ordered list of predicted masks, length N_max
    IoU    : 2D array of shape (M_n, P_n) with IoU[m][p]
    returns: list mIoU[1..N_max]
    """
    M_n = len(M)
    N_max = len(P_o)
    best_iou = [0.0] * M_n  # best_iou[m] = max IoU seen so far for gt‐mask m
    sum_best = 0.0  # sum of best_iou over all m
    mIoUs = [0.0] * N_max

    for N in range(1, N_max + 1):
        p_idx = P_o[N - 1]  # the Nth prediction
        # update each ground‐truth mask’s best IoU
        for m in range(M_n):
            new_iou = IoU[m][p_idx]
            if new_iou > best_iou[m]:
                sum_best += new_iou - best_iou[m]
                best_iou[m] = new_iou
        # compute mIoU for first N predictions
        mIoUs[N - 1] = sum_best / M_n

    return mIoUs


def oracle_sequence(M, P, IoU):
    """
    M   : ground‐truth list, |M|
    P   : list of all predictions, |P|
    IoU : 2D array IoU[m][p]
    returns: P_o* (list of p‐indices), and mIoU‐curve[1..|P|]
    """
    M_n = len(M)
    P_set = set(P)
    best_iou = [0.0] * M_n
    seq = []
    mIoUs = []
    sum_best = 0.0

    while P_set:
        # find p with max marginal gain
        best_p, best_gain = None, -1.0
        for p in P_set:
            gain = 0.0
            for m in range(M_n):
                delta = IoU[m][p] - best_iou[m]
                if delta > 0:
                    gain += delta
            if gain > best_gain:
                best_gain, best_p = gain, p

        # pick it
        P_set.remove(best_p)
        seq.append(best_p)
        # update best_iou & sum_best
        for m in range(M_n):
            if IoU[m][best_p] > best_iou[m]:
                sum_best += IoU[m][best_p] - best_iou[m]
                best_iou[m] = IoU[m][best_p]

        # record mIoU after adding best_p
        mIoUs.append(sum_best / M_n)

    return seq, mIoUs


import matplotlib.patches as mpatches

def visualize_sample(image, gt_masks, pred_masks, iou_mat, sample_idx, out_dir="viz_samples"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    # 1. Input image
    axs[0].imshow(image)
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    # 2. GT masks overlay
    gt_overlay = image.copy()
    for i, mask in enumerate(gt_masks):
        color = np.random.rand(3,)
        gt_overlay = gt_overlay.copy()
        gt_overlay[mask > 0] = 0.5 * gt_overlay[mask > 0] + 0.5 * color * 255
        axs[1].imshow(gt_overlay)
    axs[1].set_title("GT Masks Overlay")
    axs[1].axis('off')

    # 3. Pred masks overlay
    pred_overlay = image.copy()
    for i, mask in enumerate(pred_masks):
        color = np.random.rand(3,)
        pred_overlay = pred_overlay.copy()
        pred_overlay[mask > 0] = 0.5 * pred_overlay[mask > 0] + 0.5 * color * 255
        axs[2].imshow(pred_overlay)
    axs[2].set_title("Predicted Masks Overlay")
    axs[2].axis('off')

    # 4. IoU matrix heatmap
    im = axs[3].imshow(iou_mat, cmap='viridis', vmin=0, vmax=1)
    axs[3].set_title("IoU Matrix (GT x Pred)")
    axs[3].set_xlabel("Pred Mask Index")
    axs[3].set_ylabel("GT Mask Index")
    plt.colorbar(im, ax=axs[3])

    plt.tight_layout()
    plt.savefig(f"{out_dir}/sample_{sample_idx:04d}.png")
    plt.close(fig)

import matplotlib.colors as mcolors

def visualize_gt_iou_overlay(image, gt_masks, iou_mat, sample_idx, out_dir="viz_samples"):
    """
    For each GT mask, overlay it with a color corresponding to its best IoU.
    Red = 0, Green = 1, Yellow = 0.5.
    Saves two images: with and without the image background.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    # Compute best IoU for each GT mask
    best_ious = iou_mat.max(axis=1)  # (num_gt,)
    # Create color map: 0=red, 0.5=yellow, 1=green
    cmap = mcolors.LinearSegmentedColormap.from_list("iou_cmap", ["red", "yellow", "green"])
    # Overlay on blank
    overlay = np.zeros((*gt_masks[0].shape, 3), dtype=np.float32)
    for i, (mask, iou) in enumerate(zip(gt_masks, best_ious)):
        color = np.array(cmap(iou)[:3])  # RGB
        mask = mask.astype(bool)
        overlay[mask] = color
    plt.imsave(f"{out_dir}/sample_{sample_idx:04d}_gt_iou_overlay.png", overlay)
    # Overlay on image
    img_norm = image.astype(np.float32) / 255.0
    alpha = 0.5
    overlay_on_img = img_norm.copy()
    mask_any = np.zeros(mask.shape, dtype=bool)
    for i, (mask, iou) in enumerate(zip(gt_masks, best_ious)):
        color = np.array(cmap(iou)[:3])
        mask = mask.astype(bool)
        mask_any |= mask
        overlay_on_img[mask] = (1 - alpha) * overlay_on_img[mask] + alpha * color
    # Only blend where there is a mask
    plt.imsave(f"{out_dir}/sample_{sample_idx:04d}_gt_iou_overlay_on_img.png", overlay_on_img)

##### SEGNEXT ############
def load_config_file(config_path, model_name=None, return_edict=False):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "SUBCONFIGS" in cfg:
        if model_name is not None and model_name in cfg["SUBCONFIGS"]:
            cfg.update(cfg["SUBCONFIGS"][model_name])
        del cfg["SUBCONFIGS"]

    return edict(cfg) if return_edict else cfg

def generate_masks_segnext(image, mask_generator: BasePredictor):
    probs = mask_generator.predict_sat(points_per_side=mask_generator.points_per_side)
    probs = np.array(probs)
    logits = np.log(probs / (1-probs))
    scores = calculate_stability_score(logits, 0, 1)
    probs = probs[np.argsort(scores)]
    scores = scores[np.argsort(scores)]
    kept_masks, kept_indices = nms_masks(probs > 0.5, scores, iou_threshold=mask_generator.nms_thresh)
    kept_scores = scores[kept_indices]
    assert (kept_scores == np.sort(kept_scores)[::-1]).all()
    if (kept_scores > mask_generator.stability_thresh).any():
        kept_masks = kept_masks[kept_scores > mask_generator.stability_thresh]
    else:
        print("stability threshold too high for this sample, wouldn't get any masks")
    return kept_masks


def calculate_stability_score(
    logits: torch.Tensor, mask_threshold: float, threshold_offset: float
) -> torch.Tensor:
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    # One mask is always contained inside the other.
    # Save memory by preventing unnecessary cast to torch.int64
    intersections = (logits > (mask_threshold + threshold_offset)).sum(-1).sum(-1)
    unions = (logits > (mask_threshold - threshold_offset)).sum(-1).sum(-1)
    return intersections / unions


def nms_masks(masks, scores, iou_threshold=0.97):
    """
    Perform Non-Maximum Suppression (NMS) on masks based on their scores.

    Args:
        masks (np.ndarray): Array of masks with shape (M, H, W).
        scores (np.ndarray): Array of scores with shape (M,).
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        np.ndarray: Array of masks after NMS with shape (K, H, W).
    """
    M = masks.shape[0]
    mask_shape = masks.shape[1:]  # (H, W) or any shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert masks and scores to torch tensors
    mf = torch.from_numpy(masks.reshape(M, -1)).float().to(device)
    scores = torch.from_numpy(scores).to(device)

    # Sort masks by scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    mf = mf[sorted_indices]

    keep = []
    original_indices = []
    while mf.shape[0] > 0:
        # Select the mask with the highest score
        current_mask = mf[0]
        keep.append(current_mask.cpu())
        original_indices.append(sorted_indices[0].item())

        if mf.shape[0] == 1:
            break

        # Compute IoU of the current mask with the rest
        inters = torch.matmul(mf[1:], current_mask)
        areas = mf[1:].sum(dim=1) + current_mask.sum() - inters
        ious = inters / areas

        # Keep masks with IoU less than the threshold
        mask_to_keep = ious < iou_threshold
        mf = mf[1:][mask_to_keep]
        sorted_indices = sorted_indices[1:][mask_to_keep]

    # Convert kept masks back to original shape
    kept_masks = torch.stack(keep).reshape(-1, *mask_shape).cpu().numpy()
    return kept_masks > 0, original_indices




######## SAM2 ###############
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator, rle_to_mask
import numpy as np
from PIL import Image

def get_mask_generator(
    points_per_side,
    stability_score_thresh,
    box_nms_thresh,
    pred_iou_thresh,
    points_per_batch,
    crop_nms_thresh,
    crop_n_layers,
    device,
):
    sam2_model = build_sam2(
        config_file="configs/sam2.1/sam2.1_hiera_t.yaml",
        ckpt_path="checkpoints/sam2.1_hiera_tiny.pt",
        device=device,
        mode="eval",
        apply_postprocessing=False,
    )
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=points_per_side,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,
        pred_iou_thresh=pred_iou_thresh,
        points_per_batch=points_per_batch,
        crop_nms_thresh=crop_nms_thresh,
        crop_n_layers=crop_n_layers,
    )
    return mask_generator


def generate_masks_sam(image, mask_generator, multiscale=True):
    mask_data = mask_generator._generate_masks(image)
    stability_scores = mask_data["stability_score"]
    # logits = mask_data["low_res_masks"]
    ious = mask_data["iou_preds"]
    mask_data["segmentations"] = [
        rle_to_mask(rle) for rle in mask_data["rles"]
    ]  # masks
    # select one mask per point (the one with biggest score)
    scores = (ious + stability_scores) / 2
    if not multiscale:
        points = np.array(mask_data["points"])
        unique_points = np.unique(points, axis=0)
        keep_indices = []
        for point in unique_points:
            same_point_mask = np.all(point == points, axis=1)
            point_scores = scores[same_point_mask]
            max_score = point_scores.max()
            mask_indices = np.where(same_point_mask)[0]
            max_score_indices = mask_indices[point_scores==max_score]
            keep_indices.extend(max_score_indices.tolist())
        # Filter masks and sort by score
        keep_indices = np.array(keep_indices)
        masks = np.array(mask_data["segmentations"])[keep_indices]
        scores = scores[keep_indices]
    else:
        masks = np.array(mask_data["segmentations"])
    sorted_order = np.argsort(scores)
    sorted_masks = masks[sorted_order]
    return sorted_masks#, logits, ious, stability_scores, points

##### MAIN ################
def main(checkpoint, tag, datasets="DAVIS,HQSeg44K,Hypersim", device="cuda", vis=False, points_per_side=32, nms_thresh=0.75, stability_thresh=0.0):
    cfg = load_config_file("config.yml", return_edict=True)
    if "cuda" in device:
        device = (
            torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        )
    else:
        device = torch.device("cpu")
    logs_path = Path("logs")
    logs_path.mkdir(exist_ok=True, parents=True)

    is_sam = "sam" in checkpoint
    if is_sam:
        assert checkpoint.split("/")[-1] == "sam2.1_hiera_base_plus.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        sam2_model = build_sam2(model_cfg, checkpoint)
        predictor = SAM2AutomaticMaskGenerator(sam2_model,
            points_per_side=points_per_side,
            stability_score_thresh=stability_thresh,
            box_nms_thresh=nms_thresh,
            pred_iou_thresh=0,
            # points_per_batch=points_per_batch,
            # crop_nms_thresh=crop_nms_thresh,
            crop_n_layers=0,
            device=device,
        )
        mask_generator = generate_masks_sam
    else:
        ckpt_path = Path(checkpoint)
        model = load_is_model(ckpt_path, device)
        predictor = BasePredictor(model)
        predictor.points_per_side = points_per_side
        predictor.nms_thresh = nms_thresh
        predictor.stability_thresh = stability_thresh
        mask_generator = generate_masks_segnext

    datasets = datasets if isinstance(datasets, tuple) else datasets.split(",")
    for dataset_name in datasets:
        if dataset_name.lower() == "davis":
            dataset = DavisDataset(cfg.DAVIS_PATH)
        elif dataset_name.lower() == "hqseg44k":
            dataset = HQSeg44kDataset(cfg.HQSeg44K_PATH, split="val")
        elif dataset_name.lower() == "hypersim":
            dataset = HypersimDataset(cfg.Hypersim_PATH)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")


    all_curves = []
    st = time.time()
    for index in tqdm(range(len(dataset)), leave=False, disable=True):
        sample = dataset.get_sample(index)
        image = np.array(sample.image.convert("RGB"))
        print(f'processing index {index} with shape {image.shape}, avg speed {index/(time.time()-st):.3f} samples/sec')

        with torch.no_grad():
            if not is_sam:
                predictor.set_image(image)
                predictor.prev_mask = torch.zeros_like(predictor.prev_mask)  # make sure there's no interfearence
            pred_masks = mask_generator(image, predictor)
            if isinstance(pred_masks, list):
                pred_masks = np.array(pred_masks)

            # Prepare GT masks
            gt_masks = [sample.gt_mask(obj_id) for obj_id in sample.objects_ids]
            gt_masks = [m for m in gt_masks if m.sum() > 256]
            if len(gt_masks) < 1:
                continue
            gt_masks_arr = np.stack(gt_masks).astype(np.float32)
            pred_masks_arr = pred_masks.astype(np.float32)

            # Compute IoU matrix (M, N)
            gt_torch = torch.from_numpy(gt_masks_arr).to(device)
            pred_torch = torch.from_numpy(pred_masks_arr).to(gt_torch.device)
            iou_mat = iou_one_loop(gt_torch, pred_torch).cpu().numpy()

            if vis:
                visualize_sample(
                    image,
                    gt_masks_arr,
                    pred_masks_arr,
                    iou_mat,
                    sample_idx=index
                )
                visualize_gt_iou_overlay(image, gt_masks_arr, iou_mat, index)

            # compute miou vs number of masks curve
            M = list(range(gt_masks_arr.shape[0]))
            P_o = list(range(pred_masks_arr.shape[0]))  # model output order
            miou_curve = np.array(incremental_mIoU(M, P_o, iou_mat))
            all_curves.append(miou_curve)           
            with open(f'all_curves_{tag}.pickle', 'wb') as f:
                pickle.dump(all_curves, f)

            gc.collect()
            torch.cuda.empty_cache()

    # After the loop, aggregate and plot/print results
    if all_curves and vis:
        max_len = max(len(curve) for curve in all_curves)
        curves_padded = [np.pad(curve, (0, max_len - len(curve)), constant_values=np.nan) for curve in all_curves]
        curves_arr = np.stack(curves_padded)
        mean_curve = np.nanmean(curves_arr, axis=0)
        std_curve = np.nanstd(curves_arr, axis=0)
        print("Mean mIoU curve (model order):", mean_curve)
        plt.figure()
        plt.plot(np.arange(1, len(mean_curve)+1), mean_curve, label="Mean mIoU")
        plt.fill_between(np.arange(1, len(mean_curve)+1), mean_curve-std_curve, mean_curve+std_curve, alpha=0.3, label="Std Dev")
        plt.xlabel("Number of masks")
        plt.ylabel("Mean IoU (model order)")
        plt.title("Zero-shot mIoU vs. number of masks (model order)")
        plt.grid()
        plt.legend()
        plt.savefig("viz_samples/mean_miou_curve.png")

        # Histogram of best IoUs per GT mask
        best_ious = []
        for curve in curves_arr:
            best_ious.extend(curve[~np.isnan(curve)])
        plt.figure()
        plt.hist(best_ious, bins=20, range=(0,1))
        plt.xlabel("Best IoU per GT mask")
        plt.ylabel("Count")
        plt.title("Distribution of Best IoUs")
        plt.savefig("viz_samples/best_iou_histogram.png")                                             

    
if __name__ == '__main__':
    from fire import Fire
    Fire(main)