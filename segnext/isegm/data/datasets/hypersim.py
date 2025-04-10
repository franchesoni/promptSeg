from pathlib import Path
import numpy as np
import h5py
from PIL import Image
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class HypersimDataset(ISDataset):
    def __init__(self, dataset_path, **kwargs):
        super(HypersimDataset, self).__init__(**kwargs)
        self.dataset_path = Path(dataset_path)

        print("Finding images and masks...")
        images = sorted(
            self.dataset_path.glob(
                "ai_*/images/scene_cam_00_final_preview/frame.0000.color.jpg"
            )
        )
        masks = sorted(
            self.dataset_path.glob(
                "ai_*/images/scene_cam_00_geometry_hdf5/frame.0000.render_entity_id.hdf5"
            )
        )
        self.images, self.masks = [], []
        for mask_path in masks:
            img_path = Path(
                str(mask_path).split("_geometry")[0]
                + "_final_preview/frame.0000.color.jpg"
            )
            if img_path in images:
                self.images.append(img_path)
                self.masks.append(mask_path)
        print("Found {} images and masks.".format(len(self.images)))

    def get_sample(self, index) -> DSample:
        image = Image.open(self.images[index]).convert("RGB")
        labels = h5py.File(self.masks[index], "r")["dataset"][:]
        objects_ids = np.unique(labels)

        return DSample(image, labels, objects_ids=objects_ids, sample_id=index)
