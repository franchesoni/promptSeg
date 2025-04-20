import numpy as np
import sys
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from albumentations import (
    Compose,
    ShiftScaleRotate,
    RandomRotate90,
    RandomBrightnessContrast,
    PadIfNeeded,
    RGBShift,
)

PYTHON_VERSION = sys.version[:4]
if PYTHON_VERSION == "3.13":
    from albumentations import (
        HorizontalFlip,
        VerticalFlip,
        OneOf,
        LongestMaxSize,
        RandomScale,
    )
else:
    raise RuntimeError("should use python 3.13")

from isegm.model.is_plainvit_model import PlainVitModel
from isegm.data.points_sampler import MultiPointSampler
from isegm.data.datasets import CocoLvisDataset
from isegm.model.losses import NormalizedFocalLossSigmoid
from isegm.utils.misc import save_checkpoint

MODEL_NAME = "ours"


def main(cfg):
    from sam2.build_sam import build_sam2

    model_cfg = "configs/sam2.1/sam2.1_hiera_b+_512.yaml"
    checkpoint = (
        "/home/fmarchesoni/promptSeg/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    )
    model = build_sam2(model_cfg, checkpoint, device=cfg.device)
    # model = build_sam2(model_cfg, device=cfg.device)
    # checkpoint = (
    #     "/home/fmarchesoni/promptSeg/sam2/checkpoints/mae_hiera_base_plus_224.pth"
    # )
    # hiera_state_dict0 = torch.load(checkpoint)["model_state"]
    # hiera_state_dict = {}
    # for k, v in hiera_state_dict0.items():
    #     if "pos_embed" in k:
    #         continue  # skip positional embedding
    #     if ".mlp.fc1." in k:
    #         k = k.replace(".mlp.fc1.", ".mlp.layers.0.")
    #     elif ".mlp.fc2." in k:
    #         k = k.replace(".mlp.fc2.", ".mlp.layers.1.")
    #     if k in dict(model.image_encoder.trunk.named_parameters()):
    #         hiera_state_dict[k] = v
    # model.image_encoder.trunk.load_state_dict(hiera_state_dict, strict=False)
    train(model, cfg)


def train(model: PlainVitModel, cfg, num_epochs=21) -> None:
    cfg.img_size = model.image_size
    cfg.val_batch_size = cfg.batch_size
    cfg.num_max_points = 1
    cfg.num_max_next_points = 0

    # initialize the model
    model.to(cfg.device)
    model.train()

    loss_fn = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)

    train_augmentator = Compose(
        [
            # RandomScale(scale_limit=(-0.25, 40)),
            (
                OneOf(
                    [
                        HorizontalFlip(),
                        VerticalFlip(),
                        Compose([HorizontalFlip(), VerticalFlip()]),
                    ],
                    p=0.5,
                )
                if PYTHON_VERSION == "3.13"
                else Flip()
            ),
            RandomRotate90(),
            # ShiftScaleRotate(
            #     shift_limit=0.03,
            #     scale_limit=0,
            #     rotate_limit=(-3, 3),
            #     border_mode=0,
            #     p=0.75
            # ),
            RandomBrightnessContrast(
                brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75
            ),
            RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75),
            LongestMaxSize(max_size=cfg.img_size),
            PadIfNeeded(
                min_height=cfg.img_size,
                min_width=cfg.img_size,
                border_mode=0,
                position="top_left",
                **({} if PYTHON_VERSION == "3.13" else {"value": 0}),
            ),
        ],
        p=1.0,
        seed=cfg.seed,
    )

    # points_sampler = MultiPointSampler(
    #     one_random_click_sampler="posneg",
    #     max_num_points=cfg.num_max_points,
    #     prob_gamma=0.80,
    #     merge_objects_prob=0,
    #     max_num_merged_objects=1,
    # )
    points_sampler = MultiPointSampler(
        one_random_click_sampler=True,
        max_num_points=cfg.num_max_points,
        prob_gamma=0.80,
        merge_objects_prob=0,
        max_num_merged_objects=1,
    )

    trainset = CocoLvisDataset(
        cfg.LVIS_v1_PATH,
        split="train",
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=30000,
        stuff_prob=0.30,
    )

    optimizer_params = {"lr": 1e-5, "betas": (0.9, 0.999), "eps": 1e-8}
    lr = optimizer_params["lr"]
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 90], gamma=0.1
    )

    import random

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_dataloader = torch.utils.data.DataLoader(
        trainset,
        cfg.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=cfg.workers,
    )

    writer = SummaryWriter()
    for epoch in range(num_epochs):
        writer.add_scalar(
            "learning_rate",
            optimizer.param_groups[0]["lr"],
            epoch * len(train_dataloader),
        )
        for i, batch_data in enumerate(train_dataloader):
            # loss = batch_forward(batch_data)
            batch_data = {k: v.to(cfg.device) for k, v in batch_data.items()}
            image, gt_mask = batch_data["images"], batch_data["instances"]
            points = batch_data["points"]

            # exctract image features
            backbone_out = model.forward_image(image)
            backbone_out, vision_feats, vision_pos_embeds, feat_sizes = (
                model._prepare_backbone_features(backbone_out)
            )
            B, C = image.size(0), model.hidden_dim
            Hf, Wf = feat_sizes[-1]
            pix_feat = vision_feats[-1].permute(1, 2, 0).view(B, C, Hf, Wf)
            if len(vision_feats) > 1:
                high_res_features = []
                for feat, (h, w) in zip(vision_feats[:-1], feat_sizes[:-1]):
                    # feat: [L, B, C]  →  permute → [B, C, L]
                    seq = feat.permute(1, 2, 0)
                    C = seq.size(1)
                    # now reshape to [B, C, H, W]
                    high_res_features.append(seq.view(B, C, h, w))
            else:
                high_res_features = None

            # forward
            point_coords = torch.fliplr(
                torch.stack(
                    [
                        points[b][0][:2] if points[b][0].sum() else points[b][1][:2]
                        for b in range(len(points))
                    ]
                )
            ).reshape(B, 1, 2)
            point_labels = (
                torch.tensor([1 * (points[b][0].sum()) for b in range(len(points))])
                .reshape(B, 1)
                .to(point_coords.device)
            )

            # now call _forward_sam_heads and unpack correctly:
            (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
            ) = model._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=dict(point_coords=point_coords, point_labels=point_labels),
                mask_inputs=None,
                high_res_features=high_res_features,
                multimask_output=False,
            )
            loss = torch.mean(loss_fn(high_res_multimasks, gt_mask))
            # now optim step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_item = loss.item()
            print(
                f"epoch {epoch} step {i}/{len(train_dataloader)}, loss={loss_item}",
                end="\r",
            )
            writer.add_scalar("loss", loss_item, epoch * len(train_dataloader) + i)
        writer.add_scalar(
            "learning_rate",
            optimizer.param_groups[0]["lr"],
            epoch * len(train_dataloader) + i,
        )
        lr_scheduler.step()
        # save once in a while
        if epoch % 10 == 0:
            torch.save(model.state_dict(), Path(writer.log_dir) / f"model_epoch_{epoch}.pth")
            # save_checkpoint(model, Path(writer.log_dir) / "checkpoints", epoch=epoch)
    # save_checkpoint(model, Path(writer.log_dir) / "checkpoints", epoch=999)
    torch.save(model.state_dict(), Path(writer.log_dir) / "final_model.pth")
