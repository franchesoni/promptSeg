import argparse

import yaml
import torch
from easydict import EasyDict

import torch
import segnext.models.default.clean_posneg as model_script


def main():
    # load config
    args = parse_args()
    with open("config.yml", "r") as f:
        cfg = EasyDict(yaml.safe_load(f))
    for k, v in args._get_kwargs():
        cfg[k] = v
    cfg['device'] = torch.device('cuda')


    torch.backends.cudnn.benchmark = True
    model_script.main(cfg)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-name",
        type=str,
        default="",
        help="Here you can specify the name of the experiment. "
        "It will be added as a suffix to the experiment folder.",
    )

    parser.add_argument(
        "--workers", type=int, default=0, metavar="N", help="Dataloader threads."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="You can override model batch size by specify positive number.",
    )

    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        required=False,
        help='Ids of used GPUs. You should use either this argument or "--ngpus".',
    )

    parser.add_argument(
        "--resume-exp",
        type=str,
        default=None,
        help="The prefix of the name of the experiment to be continued. "
        'If you use this field, you must specify the "--resume-prefix" argument.',
    )

    parser.add_argument(
        "--resume-prefix",
        type=str,
        default="latest",
        help="The prefix of the name of the checkpoint to be loaded.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="The number of the starting epoch from which training will continue. "
        "(it is important for correct logging and learning rate)",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Model weights will be loaded from the specified path if you use this argument.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
