
from isegm.utils.exp_imports.default import *

import os
import argparse
import importlib.util

import torch
from segnext.isegm.utils.exp import init_experiment


def main(rank):
    args = parse_args()
    model_base_name = getattr(model_script, 'MODEL_NAME', None)

    args.local_rank = rank
    cfg = init_experiment(args, model_base_name)

    torch.backends.cudnn.benchmark = True

    submain(cfg)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', type=str,
                        help='Path to the model script.')

    parser.add_argument('--exp-name', type=str, default='',
                        help='Here you can specify the name of the experiment. '
                             'It will be added as a suffix to the experiment folder.')

    parser.add_argument('--workers', type=int, default=0,
                        metavar='N', help='Dataloader threads.')

    parser.add_argument('--batch-size', type=int, default=32,
                        help='You can override model batch size by specify positive number.')

    parser.add_argument('--gpus', type=str, default='', required=False,
                        help='Ids of used GPUs. You should use either this argument or "--ngpus".')
    
    parser.add_argument('--resume-exp', type=str, default=None,
                        help='The prefix of the name of the experiment to be continued. '
                             'If you use this field, you must specify the "--resume-prefix" argument.')

    parser.add_argument('--resume-prefix', type=str, default='latest',
                        help='The prefix of the name of the checkpoint to be loaded.')

    parser.add_argument('--start-epoch', type=int, default=0,
                        help='The number of the starting epoch from which training will continue. '
                             '(it is important for correct logging and learning rate)')

    parser.add_argument('--weights', type=str, default=None,
                        help='Model weights will be loaded from the specified path if you use this argument.')

    return parser.parse_args()


def load_module(script_path):
    spec = importlib.util.spec_from_file_location("model_script", script_path)
    model_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_script)

    return model_script


if __name__ == '__main__':
    rank = int(os.environ['LOCAL_RANK'])
    main(rank)

MODEL_NAME = 'plainvit_base1024_cocolvis_sax2'


def submain(cfg):
    model = build_model(img_size=1024)
    train(model, cfg)


def build_model(img_size) -> PlainVitModel:
    backbone_params = dict(
        img_size=(img_size, img_size),
        patch_size=(16,16),
        in_chans=3,
        embed_dim=768,
        depth=12,
        global_atten_freq=3,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
    )

    neck_params = dict(in_dim = 768, out_dims = [128, 256, 512, 1024],)

    head_params = dict(
        in_channels=[128, 256, 512, 1024],
        in_select_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=1,
        out_channels=256,
    )

    fusion_params = dict(
        type='self_attention',
        depth=2,
        params=dict(dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True,)
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


def train(model: PlainVitModel, cfg) -> None:
    cfg.img_size = model.backbone.patch_embed.img_size[0]
    cfg.val_batch_size = cfg.batch_size
    cfg.num_max_points = 1
    cfg.num_max_next_points = 0

    # initialize the model
    model.backbone.init_weights_from_pretrained(cfg.MAE_WEIGHTS.VIT_BASE)
    model.to(cfg.device)

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0
    cfg.loss_cfg = loss_cfg

    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.40)),
        Flip(),
        RandomRotate90(),
        ShiftScaleRotate(
            shift_limit=0.03, 
            scale_limit=0,
            rotate_limit=(-3, 3), 
            border_mode=0, 
            p=0.75
        ),
        RandomBrightnessContrast(
            brightness_limit=(-0.25, 0.25),
            contrast_limit=(-0.15, 0.4), 
            p=0.75
        ),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75),
        ResizeLongestSide(target_length=cfg.img_size),
        PadIfNeeded(
            min_height=cfg.img_size,
            min_width=cfg.img_size,
            border_mode=0,
            value=0,
            position='top_left',
        ),
    ], p=1.0)

    val_augmentator = Compose([
        ResizeLongestSide(target_length=cfg.img_size),
        PadIfNeeded(
            min_height=cfg.img_size, 
            min_width=cfg.img_size, 
            border_mode=0,
            value=0,
            position='top_left',
        ),
    ], p=1.0)

    points_sampler = MultiPointSampler(
        cfg.num_max_points, 
        prob_gamma=0.80,
        merge_objects_prob=0.15,
        max_num_merged_objects=2
    )

    trainset = CocoLvisDataset(
        cfg.LVIS_v1_PATH,
        split='train',
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=30000,
        stuff_prob=0.30
    )

    valset = CocoLvisDataset(
        cfg.LVIS_v1_PATH,
        split='val',
        augmentator=val_augmentator,
        min_object_area=1000,
        points_sampler=points_sampler,
        epoch_len=2000
    )

    optimizer_params = {'lr': 5e-5, 'betas': (0.9, 0.999), 'eps': 1e-8}
    lr_scheduler = partial(
        torch.optim.lr_scheduler.MultiStepLR, milestones=[50, 90], gamma=0.1
    )
    trainer = ISTrainer(
        model, 
        cfg,
        trainset, 
        valset,
        optimizer='adam',
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        checkpoint_interval=[(0, 10), (90, 1)],
        image_dump_interval=500,
        metrics=[AdaptiveIoU()],
        max_interactive_points=cfg.num_max_points,
        max_num_next_clicks=cfg.num_max_next_points
    )
    trainer.run(num_epochs=100, validation=False)
