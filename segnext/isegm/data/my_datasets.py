from pathlib import Path

import cv2
import numpy as np
import random
import pickle
import numpy as np
import torch
from torchvision import transforms

from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class ISDataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        augmentator=None,
        points_sampler=None,
        min_object_area=0,
        keep_background_prob=0.0,
        with_image_info=False,
        samples_scores_path=None,
        samples_scores_gamma=1.0,
        epoch_len=-1,
    ):
        super(ISDataset, self).__init__()
        self.epoch_len = epoch_len
        self.augmentator = augmentator
        self.min_object_area = min_object_area
        self.keep_background_prob = keep_background_prob
        self.points_sampler = points_sampler
        self.with_image_info = with_image_info
        self.samples_precomputed_scores = self._load_samples_scores(
            samples_scores_path, samples_scores_gamma
        )
        self.to_tensor = transforms.ToTensor()

        self.dataset_samples = None

    def __getitem__(self, index):
        if self.samples_precomputed_scores is not None:
            index = np.random.choice(
                self.samples_precomputed_scores["indices"],
                p=self.samples_precomputed_scores["probs"],
            )
        else:
            if self.epoch_len > 0:
                index = random.randrange(0, len(self.dataset_samples))

        sample = self.get_sample(index)
        sample = self.augment_sample(sample)
        sample.remove_small_objects(self.min_object_area)

        self.points_sampler.sample_object(sample)
        points = np.array(self.points_sampler.sample_points())
        mask = self.points_sampler.selected_mask

        output = {
            "images": self.to_tensor(sample.image),
            "points": points.astype(np.float32),
            "instances": mask,
        }

        if self.with_image_info:
            output["image_info"] = sample.sample_id

        return output

    def augment_sample(self, sample) -> DSample:
        if self.augmentator is None:
            return sample

        valid_augmentation = False
        while not valid_augmentation:
            sample.augment(self.augmentator)
            keep_sample = (
                self.keep_background_prob < 0.0
                or random.random() < self.keep_background_prob
            )
            valid_augmentation = len(sample) > 0 or keep_sample

        return sample

    def get_sample(self, index) -> DSample:
        raise NotImplementedError

    def __len__(self):
        if self.epoch_len > 0:
            return self.epoch_len
        else:
            return self.get_samples_number()

    def get_samples_number(self):
        return len(self.dataset_samples)

    @staticmethod
    def _load_samples_scores(samples_scores_path, samples_scores_gamma):
        if samples_scores_path is None:
            return None

        with open(samples_scores_path, "rb") as f:
            images_scores = pickle.load(f)

        probs = np.array([(1.0 - x[2]) ** samples_scores_gamma for x in images_scores])
        probs /= probs.sum()
        samples_scores = {"indices": [x[0] for x in images_scores], "probs": probs}
        print(f"Loaded {len(probs)} weights with gamma={samples_scores_gamma}")
        return samples_scores


class HQSeg44kDataset(ISDataset):
    def __init__(self, dataset_path, split="train", **kwargs) -> None:
        super(HQSeg44kDataset, self).__init__(**kwargs)
        self.dataset_path = Path(dataset_path)

        # train set: 44320 images
        dis_train = {
            "name": "DIS5K-TR",
            "im_dir": "DIS5K/DIS-TR/im",
            "gt_dir": "DIS5K/DIS-TR/gt",
        }
        thin_train = {
            "name": "ThinObject5k-TR",
            "im_dir": "thin_object_detection/ThinObject5K/images_train",
            "gt_dir": "thin_object_detection/ThinObject5K/masks_train",
        }
        fss_train = {
            "name": "FSS",
            "im_dir": "cascade_psp/fss_all",
            "gt_dir": "cascade_psp/fss_all",
        }
        duts_train = {
            "name": "DUTS-TR",
            "im_dir": "cascade_psp/DUTS-TR",
            "gt_dir": "cascade_psp/DUTS-TR",
        }
        duts_te_train = {
            "name": "DUTS-TE",
            "im_dir": "cascade_psp/DUTS-TE",
            "gt_dir": "cascade_psp/DUTS-TE",
        }
        ecssd_train = {
            "name": "ECSSD",
            "im_dir": "cascade_psp/ecssd",
            "gt_dir": "cascade_psp/ecssd",
        }
        msra_train = {
            "name": "MSRA10K",
            "im_dir": "cascade_psp/MSRA_10K",
            "gt_dir": "cascade_psp/MSRA_10K",
        }

        # valid set: 1537 images
        dis_val = {
            "name": "DIS5K-VD",
            "im_dir": "DIS5K/DIS-VD/im",
            "gt_dir": "DIS5K/DIS-VD/gt",
        }
        thin_val = {
            "name": "ThinObject5k-TE",
            "im_dir": "thin_object_detection/ThinObject5K/images_test",
            "gt_dir": "thin_object_detection/ThinObject5K/masks_test",
        }
        coift_val = {
            "name": "COIFT",
            "im_dir": "thin_object_detection/COIFT/images",
            "gt_dir": "thin_object_detection/COIFT/masks",
        }
        hrsod_val = {
            "name": "HRSOD",
            "im_dir": "thin_object_detection/HRSOD/images",
            "gt_dir": "thin_object_detection/HRSOD/masks_max255",
        }

        if split == "train":
            self.datasets = [
                dis_train,
                thin_train,
                fss_train,
                duts_train,
                duts_te_train,
                ecssd_train,
                msra_train,
            ]
        elif split == "val":
            self.datasets = [dis_val, thin_val, coift_val, hrsod_val]
        else:
            raise ValueError(f"Undefined split: {split}")

        self.dataset_samples = []
        for idx, dataset in enumerate(self.datasets):
            image_path = self.dataset_path / dataset["im_dir"]
            samples = [(x.stem, idx) for x in sorted(image_path.glob("*.jpg"))]
            self.dataset_samples.extend(samples)

        assert len(self.dataset_samples) > 0

    def get_sample(self, index) -> DSample:
        image_name, idx = self.dataset_samples[index]
        image_path = str(
            self.dataset_path / self.datasets[idx]["im_dir"] / f"{image_name}.jpg"
        )
        mask_path = str(
            self.dataset_path / self.datasets[idx]["gt_dir"] / f"{image_name}.png"
        )

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = np.max(cv2.imread(mask_path).astype(np.int32), axis=2)
        instances_mask[instances_mask > 0] = 1

        return DSample(image, instances_mask, objects_ids=[1], sample_id=index)


class DavisDataset(ISDataset):
    def __init__(
        self, dataset_path, images_dir_name="img", masks_dir_name="gt", **kwargs
    ):
        super(DavisDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob("*.*"))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob("*.*")}

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split(".")[0]])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = np.max(cv2.imread(mask_path).astype(np.int32), axis=2)
        instances_mask[instances_mask > 0] = 1

        return DSample(image, instances_mask, objects_ids=[1], sample_id=index)
