import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import yaml
from easydict import EasyDict as edict
import matplotlib.pyplot as plt

segnext_path = Path(__file__).parent.parent.as_posix()
sys.path.insert(0, segnext_path)

from isegm.data.my_datasets import DavisDataset, HQSeg44kDataset, HypersimDataset
from isegm.model.is_plainvit_model import PlainVitModel
from isegm.inference.predictor import BasePredictor
from isegm.inference.utils import load_is_model, get_iou
from isegm.inference.clicker import RandomClicker as Clicker, Click


def load_config_file(config_path, model_name=None, return_edict=False):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "SUBCONFIGS" in cfg:
        if model_name is not None and model_name in cfg["SUBCONFIGS"]:
            cfg.update(cfg["SUBCONFIGS"][model_name])
        del cfg["SUBCONFIGS"]

    return edict(cfg) if return_edict else cfg


def visualize(clicker, image, gt_mask, pred_probs, dataset_name, index):
    # Visualization code to save images as png
    import matplotlib.pyplot as plt
    import os

    # Create visualization directory if it doesn't exist
    vis_dir = Path("visualization")
    vis_dir.mkdir(exist_ok=True, parents=True)

    # Get click coordinates
    click = clicker.get_clicks()[0]
    click_y, click_x = click.coords_and_indx[0], click.coords_and_indx[1]
    click_type = "Positive" if click.is_positive else "Negative"

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Image with click point
    axes[0].imshow(image)
    axes[0].plot(click_x, click_y, "ro", markersize=10)  # Red circle for click
    axes[0].set_title(f"{click_type} Click at ({click_x}, {click_y})")
    axes[0].axis("off")

    # pred vs. gt
    err_map = np.zeros_like(image, dtype=float)
    err_map[:, :, 0] = gt_mask if pred_probs is None else pred_probs
    err_map[:, :, 1] = gt_mask
    err_map[:, :, 2] = gt_mask
    axes[1].imshow(err_map)
    axes[1].set_title(
        "Error Map (red=pred, blue=green=gt_mask)"
        if (pred_probs is not None)
        else "Ground Truth Mask"
    )
    axes[1].axis("off")

    # Ground truth mask
    axes[2].imshow(gt_mask, cmap="gray")
    axes[2].set_title("Ground Truth Mask")
    axes[2].axis("off")

    # Save the visualization
    sample_name = f"{dataset_name}_{index}"
    fig.suptitle(sample_name)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"{sample_name}.png"), dpi=150)
    plt.close(fig)


def flip_rotate_image(image, rotate, flip):
    out = image.copy()
    if flip == 1:
        out = np.fliplr(out)
    if rotate > 0:
        out = np.rot90(out, k=rotate)
    return out.copy()


def flip_rotate_click(click, height, width, rotate, flip):
    """
    Transform click coordinates according to rotation and flipping.

    Args:
        click: Click object with coords_and_indx attribute
        height: Height of the image (needed for coordinate transformation)
        width: Width of the image (needed for coordinate transformation)
        rotate: Number of 90-degree rotations (0-3)
        flip: Whether to flip horizontally (0: no flip, 1: flip)

    Returns:
        A new click with transformed coordinates
    """
    # Extract coordinates
    y, x, indx = click.coords_and_indx

    # copy original dimensions
    orig_height, orig_width = height, width

    # Apply horizontal flip first (if requested)
    if flip == 1:
        x = width - x - 1  # Flip x-coordinate

    # Apply rotation
    for _ in range(rotate):
        # 90-degree rotation: (y, x) -> (x, height-y-1)
        y, x = width - x - 1, y
        # After rotation, height and width are swapped
        height, width = width, height

    # Update coordinates in the new click object
    new_click = Click(is_positive=click.is_positive, coords=(y, x), indx=indx)
    return new_click


def main(checkpoint, datasets="DAVIS,HQSeg44K,Hypersim", device="cuda", vis=False, c=1, aug=False):
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
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        assert checkpoint.split("/")[-1] == "sam2.1_hiera_base_plus.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    else:
        ckpt_path = Path(checkpoint)
        model = load_is_model(ckpt_path, device)
        # model.load_state_dict(torch.load('segnext/epoch_9.pth', map_location=device))
        predictor = BasePredictor(model)

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

        # evaluate dataset
        all_ious = []
        for index in tqdm(range(len(dataset)), leave=False):
            sample = dataset.get_sample(index)
            image = sample.image

            with torch.no_grad():
                predictor.set_image(image)
                sample_ious = []

                for object_id in sample.objects_ids:
                    # evaluate sample
                    gt_mask = sample.gt_mask(object_id)
                    if gt_mask.sum() < 256:
                        continue  # skip very small masks (16x16)
                    iou_per_click_indx = []
                    for click_indx in range(c):
                        clicker = Clicker(gt_mask=gt_mask, seed=click_indx)
                        pred_mask = np.zeros_like(gt_mask)
                        clicker.make_next_click(pred_mask)

                        if is_sam:
                            point_coords = np.fliplr(
                                np.array(clicker.get_clicks()[0].coords).reshape(1, 2)
                            ).copy()
                            point_labels = np.array([1])
                            masks, ious, logits = predictor.predict(
                                point_coords, point_labels, multimask_output=True
                            )
                            pred_mask = masks[np.argmax(ious)] > 0.5
                            pred_probs = pred_mask  # we don't really have the full resolution probability maps here
                        else:
                            predictor.prev_mask = torch.zeros_like(predictor.prev_mask)  # always reset prev mask before prediction
                            pred_probs = predictor.predict(clicker)
                            pred_mask = pred_probs > 0.5

                        iou = get_iou(gt_mask, pred_mask)
                        iou_per_click_indx.append(iou)

                    sample_ious.append(np.mean(iou_per_click_indx))
                    if vis:
                        visualize(
                            clicker, image, gt_mask, pred_probs, dataset_name, index
                        )

            all_ious.append(np.mean(sample_ious))

        print("mean iou for", dataset_name, np.mean(all_ious))


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
