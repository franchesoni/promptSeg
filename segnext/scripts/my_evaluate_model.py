import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import yaml
from easydict import EasyDict as edict

segnext_path = Path(__file__).parent.parent.as_posix()
sys.path.insert(0, segnext_path)

from isegm.data.my_datasets import DavisDataset, HQSeg44kDataset
from isegm.model.is_plainvit_model import PlainVitModel
from isegm.inference.predictor import BasePredictor
from isegm.inference.utils import load_is_model, get_iou
from isegm.inference.clicker import RandomClicker as Clicker


def load_config_file(config_path, model_name=None, return_edict=False):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "SUBCONFIGS" in cfg:
        if model_name is not None and model_name in cfg["SUBCONFIGS"]:
            cfg.update(cfg["SUBCONFIGS"][model_name])
        del cfg["SUBCONFIGS"]

    return edict(cfg) if return_edict else cfg


def main(checkpoint, datasets="DAVIS,HQSeg44K", cpu=False):
    cfg = load_config_file("config.yml", return_edict=True)
    if not cpu:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")
    ckpt_path = Path(checkpoint)
    logs_path = Path("logs")
    logs_path.mkdir(exist_ok=True, parents=True)

    model = load_is_model(ckpt_path, device)
    # model.load_state_dict(torch.load('segnext/epoch_9.pth', map_location=device))
    predictor = BasePredictor(model)

    datasets = datasets if isinstance(datasets, tuple) else datasets.split(",")
    for dataset_name in datasets:
        if dataset_name == "DAVIS":
            dataset = DavisDataset(cfg.DAVIS_PATH)
        elif dataset_name == "HQSeg44K":
            dataset = HQSeg44kDataset(cfg.HQSeg44K_PATH, split="val")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # evaluate dataset
        all_ious = []
        for index in tqdm(range(len(dataset)), leave=False):
            sample = dataset.get_sample(index)
            image = sample.image
            assert len(sample.objects_ids) == 1

            with torch.no_grad():
                # evaluate sample
                gt_mask = sample.gt_mask(sample.objects_ids[0])
                clicker = Clicker(gt_mask=gt_mask)
                pred_mask = np.zeros_like(gt_mask)

                clicker.make_next_click(pred_mask)
                predictor.set_image(image)
                pred_probs = predictor.predict(clicker)
                pred_mask = pred_probs > 0.5

                iou = get_iou(gt_mask, pred_mask)
                all_ious.append(iou)

        print("mean iou for", dataset_name, np.mean(all_ious))


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
