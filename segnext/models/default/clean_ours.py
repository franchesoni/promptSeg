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
    model = build_model(img_size=512)
    train(model, cfg)


def build_model(img_size) -> PlainVitModel:
    backbone_params = dict(
        img_size=(img_size, img_size),
        patch_size=(16, 16),
        in_chans=3,
        embed_dim=768,
        depth=12,
        global_atten_freq=3,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
    )

    neck_params = dict(
        in_dim=768,
        out_dims=[128, 256, 512, 1024],
    )

    head_params = dict(
        in_channels=[128, 256, 512, 1024],
        in_select_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=1,
        out_channels=256,
    )

    fusion_params = dict(
        type="self_attention",
        depth=2,
        params=dict(
            dim=768,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
        ),
    )

    model = PlainVitModel(
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        fusion_params=fusion_params,
        use_disks=True,
        norm_radius=5,
    )
    return model


def train(model: PlainVitModel, cfg, num_epochs=21) -> None:
    cfg.img_size = model.backbone.patch_embed.img_size[0]
    cfg.val_batch_size = cfg.batch_size
    cfg.num_max_points = 1
    cfg.num_max_next_points = 0

    # initialize the model
    model.backbone.init_weights_from_pretrained(cfg.MAE_WEIGHTS.VIT_BASE)
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
    points_sampler = MultiPointSampler(
        one_random_click_sampler="posneg",
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

    optimizer_params = {"lr": 5e-5, "betas": (0.9, 0.999), "eps": 1e-8}
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
            image_feats = model.get_image_feats(image)
            prev_mask = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
            batch_data["points"] = points
            prompts = {"points": points, "prev_mask": prev_mask}
            prompt_feats = model.get_prompt_feats(image.shape, prompts)
            output = model(image.shape, image_feats, prompt_feats)
            loss = torch.mean(loss_fn(output["instances"], gt_mask))
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
            save_checkpoint(model, Path(writer.log_dir) / "checkpoints", epoch=epoch)
    save_checkpoint(model, Path(writer.log_dir) / "checkpoints", epoch=999)
