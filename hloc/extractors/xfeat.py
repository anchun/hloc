import glob
import os

import torch

from hloc import logger

from ..utils.base_model import BaseModel


class XFeat(BaseModel):
    default_conf = {
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        repo = "verlab/accelerated_features"
        # Reuse the local torch.hub cache if present to avoid contacting GitHub.
        # torch caches github repos under <hub_dir>/<owner>_<repo>_<ref>.
        cached_dirs = sorted(
            d
            for d in glob.glob(
                os.path.join(torch.hub.get_dir(), "verlab_accelerated_features_*")
            )
            if os.path.isdir(d)
        )
        if cached_dirs:
            logger.info(f"Loading XFeat from local torch.hub cache: {cached_dirs[0]}")
            self.net = torch.hub.load(
                cached_dirs[0],
                "XFeat",
                source="local",
                pretrained=True,
                top_k=self.conf["max_keypoints"],
            )
        else:
            self.net = torch.hub.load(
                repo,
                "XFeat",
                pretrained=True,
                top_k=self.conf["max_keypoints"],
            )
        logger.info("Load XFeat(sparse) model done.")

    def _forward(self, data):
        pred = self.net.detectAndCompute(
            data["image"], top_k=self.conf["max_keypoints"]
        )[0]
        pred = {
            "keypoints": pred["keypoints"][None],
            "keypoint_scores": pred["scores"][None],
            "descriptors": pred["descriptors"].T[None],
        }
        return pred
