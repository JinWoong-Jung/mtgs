# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import os
from typing import Tuple, Dict, List, Optional

import itertools

import cv2

from tqdm import tqdm
import subprocess as sp

from PIL import Image
import pandas as pd
import numpy as np

import torch
import torchvision.transforms.functional as TF

from mtgs.config.config_manager import ConfigManager
from mtgs.demo import HeadDetector, Tracker, GazePredictor
from mtgs.utils.utils import expand_bbox_for_demo as expand_bbox
from mtgs.utils import (
    get_social_gaze_predictions,
    get_video_creation_command,
    spatial_argmax2d,
    square_bbox,
    get_device,
    check_file,
    draw_gaze,
)

import logging

logger = logging.getLogger(__name__)


class DemoProcessor:
    def __init__(self) -> None:
        # Set configuration values
        self.cfg = ConfigManager.get_config()
        # Processing device (cpu, gpu)
        self.device = get_device(self.cfg.device)
        # Image normalization (mean and standard deviation)
        self.img_mean = list(self.cfg.image.normalization.mean)
        self.img_std = list(self.cfg.image.normalization.std)

        # Head Detector (YOLO)
        self.head_detector = HeadDetector(
            checkpoint_file=self.cfg.head_detector.checkpoint_file,
            device=self.cfg.device,
        )

        # Head tracker (OCSORT)
        self.head_tracker = Tracker(
            det_threshold=self.cfg.tracker.det_threshold,
            asso_threshold=self.cfg.tracker.asso_threshold,
            inertia=self.cfg.tracker.inertia,
            max_age=self.cfg.tracker.max_age,
        )

        # Gaze predictor (MTGS)
        self.gaze_predictor = GazePredictor(
            checkpoint_file=self.cfg.demo.checkpoint_file,
            temporal_context=self.cfg.data.temporal_context,
            image_size=self.cfg.data.image_size,
            patch_size=self.cfg.model.patch_size,
            decoder_feature_dim=self.cfg.model.decoder_feature_dim,
            decoder_use_bn=self.cfg.model.decoder_use_bn,
            device=self.cfg.device,
        )

    def get_head_bboxes(
        self, frame: np.ndarray
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return the head bounding boxes and person ids by performing head detection and tracking"""

        # Frame size
        height, width = frame.shape[:2]

        # Head detection
        raw_dets = self.head_detector.detect_heads(
            image=frame,
            image_size=self.cfg.head_detector.image_size,
            conf_thr=self.cfg.head_detector.conf_thr,
            iou_thr=self.cfg.head_detector.iou_thr,
        )

        # Processing detections
        dets = []
        for _, raw_det in enumerate(raw_dets):
            bbox, conf = raw_det[:4], raw_det[4]
            if conf > self.cfg.head_detector.detection_thr:
                bbox = expand_bbox(
                    bbox, width, height, k=self.cfg.head_detector.expand_bbox
                )
                cls_ = np.array([0.0])
                dets.append(np.concatenate([bbox, conf[None], cls_]))
        dets = np.stack(dets) if len(dets) > 0 else np.empty((0, 6))

        # Head/people tracking
        tracks = self.head_tracker.update(dets, frame)
        if len(tracks) == 0:
            return None, None

        # Person ids and head bboxes
        pids = (tracks[:, 4] - 1).astype(int)
        pids = torch.from_numpy(pids)
        head_bboxes = torch.from_numpy(tracks[:, :4]).float()

        return head_bboxes, pids

    def prepare_input(self, frame: np.ndarray, head_bboxes: torch.Tensor) -> Dict:
        """Prepare the network input (sample)"""

        # Squared head bboxes
        height, width = frame.shape[:2]
        t_head_bboxes = square_bbox(head_bboxes, width, height)

        # PIL image
        image = Image.fromarray(frame)

        # Extract and transform head images
        heads = []
        head_size = self.cfg.model.head_size
        for bbox in t_head_bboxes:
            head = TF.resize(
                TF.to_tensor(image.crop(bbox.numpy())),
                [head_size, head_size],
                antialias=True,
            )
            heads.append(head)
        heads = torch.stack(heads)
        heads = TF.normalize(heads, mean=self.img_mean, std=self.img_std)

        # Transform image
        image_size = self.cfg.data.image_size
        image = TF.to_tensor(image)
        image = TF.resize(image, [image_size, image_size], antialias=True)
        image = TF.normalize(image, mean=self.img_mean, std=self.img_std)
        image = image.unsqueeze(0)

        # Normalize head bboxes
        t_head_bboxes[:, 0] /= width
        t_head_bboxes[:, 1] /= height
        t_head_bboxes[:, 2] /= width
        t_head_bboxes[:, 3] /= height
        num_valid_people = len(t_head_bboxes)

        # Build input sample
        sample = {}
        sample["image"] = image.unsqueeze(0).to(self.device)
        sample["num_valid_people"] = num_valid_people
        # Padding
        thsize = torch.zeros((1, 3, head_size, head_size), dtype=torch.float32)
        heads = torch.cat([thsize, heads]).to(self.device)
        t_head_bboxes = torch.cat(
            [torch.zeros((1, 4), dtype=torch.float32), t_head_bboxes]
        ).to(self.device)
        sample["heads"] = heads.unsqueeze(0).unsqueeze(0).to(self.device)
        sample["head_bboxes"] = t_head_bboxes.unsqueeze(0).unsqueeze(0).to(self.device)

        return sample

    def predict(self, frame: np.ndarray) -> Dict:
        """Perform gaze prediction for each detected person in the image"""

        # Obtain the head bboxes and their personal ids by performing
        # head detection and tracking
        head_bboxes, pids = self.get_head_bboxes(frame)

        # Return empty prediction
        if head_bboxes is None and pids is None:
            pred = {}
            pred["gaze_heatmaps"] = torch.tensor([])
            pred["gaze_points"] = torch.tensor([])
            pred["gaze_vecs"] = torch.tensor([])
            pred["inouts"] = torch.tensor([])
            pred["lah"] = torch.tensor([])
            pred["laeo"] = torch.tensor([])
            pred["coatt"] = torch.tensor([])
            pred["head_bboxes"] = torch.tensor([])
            pred["pids"] = torch.tensor([])
            return pred

        # Prepare the network input (sample)
        assert head_bboxes is not None and pids is not None
        sample = self.prepare_input(frame, head_bboxes)

        # Run MTGS model (gaze prediction)
        with torch.no_grad():
            _, gaze_vecs, gaze_heatmaps, inouts, lah, laeo, coatt = (
                self.gaze_predictor.predictor(sample)
            )
            gaze_heatmaps = gaze_heatmaps.squeeze(0).squeeze(0)[1:]
            gaze_vecs = gaze_vecs.squeeze(0).squeeze(0)[1:]
            gaze_points = spatial_argmax2d(gaze_heatmaps, normalize=True)
            lah = lah.squeeze(0)  # Look at heads
            laeo = laeo.squeeze(0)  # Look at each other
            coatt = coatt.squeeze(0)  # coattention
            inouts = inouts.squeeze(0).squeeze(0)[1:]  # in-out prediction

        # Network prediction
        pred = {}
        pred["gaze_heatmaps"] = gaze_heatmaps.cpu()
        pred["gaze_points"] = gaze_points.cpu()
        pred["gaze_vecs"] = gaze_vecs.cpu()
        pred["inouts"] = inouts.sigmoid().cpu()
        pred["lah"] = lah.sigmoid().cpu()
        pred["laeo"] = laeo.sigmoid().cpu()
        pred["coatt"] = coatt.sigmoid().cpu()
        pred["head_bboxes"] = head_bboxes.cpu()
        pred["pids"] = pids.cpu()

        return pred

    def save_predictions(
        self,
        predictions: List[Dict],
        output_file: str,
        image_width: int,
        image_height: int,
    ) -> None:
        """Save gaze predictions in a pandas dataframe file (csv)"""

        # Information to save (dataframe columns)
        columns = [
            "frame_nb",  # Frame number/index
            "gaze_pt_x",  # Gaze point (x)
            "gaze_pt_y",  # Gaze point (y)
            "gaze_vec_x",  # Gaze vector (x)
            "gaze_vec_y",  # Gaze vector (y)
            "inout",  # In-Out prediction
            "lah_id",  # Look at head
            "laeo_id",  # Look at each other
            "coatt_id",  # Coattention
            "pid",  # Person id
            "xmin",  # Head box (xmin)
            "ymin",  # Head box (ymin)
            "xmax",  # Head box (xmax)
            "ymax",  # Head box (ymax)
        ]
        df = pd.DataFrame(columns=columns)

        for prediction in predictions:
            frame_nb = prediction["frame_nb"]
            num_people = len(prediction["gaze_points"])
            pair_indices = torch.tensor(
                list(itertools.permutations(torch.arange(num_people + 1), 2))
            )
            for k in range(num_people):
                # 2d gaze point
                gp_x, gp_y = prediction["gaze_points"][k].numpy()

                # Gaze vector
                gv_x, gv_y = prediction["gaze_vecs"][k].numpy()

                # In-Out prediction
                io = prediction["inouts"][k].numpy().item()

                # Person ids
                pid = prediction["pids"][k].item()

                # Normalized head bbox
                head_bbox = prediction["head_bboxes"][k].numpy()
                xmin, ymin, xmax, ymax = head_bbox
                xmin, xmax = xmin / image_width, xmax / image_width
                ymin, ymax = ymin / image_height, ymax / image_height

                # Social gaze prediction
                valid_indices = torch.where(
                    (pair_indices[:, 1] == (k + 1)).int()
                    * (pair_indices[:, 0] != 0).int()
                )[0]
                lah = prediction["lah"][0][valid_indices]
                laeo = prediction["laeo"][0][valid_indices]
                coatt = prediction["coatt"][0][valid_indices]
                # Save social gaze prediction as a dict
                social_gaze_pids = pair_indices[valid_indices][:, 0].numpy()
                lah_dict = {}
                laeo_dict = {}
                coatt_dict = {}
                for si, spid in enumerate(social_gaze_pids):
                    lah_dict[spid] = lah[si].item()
                    laeo_dict[spid] = laeo[si].item()
                    coatt_dict[spid] = coatt[si].item()

                row = {
                    "frame_nb": frame_nb,
                    "gaze_pt_x": gp_x,
                    "gaze_pt_y": gp_y,
                    "gaze_vec_x": gv_x,
                    "gaze_vec_y": gv_y,
                    "inout": io,
                    "lah_id": lah_dict,
                    "laeo_id": laeo_dict,
                    "coatt_id": coatt_dict,
                    "pid": pid,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                }
                df.loc[len(df)] = row

        df.to_csv(output_file, index=False)

    def process_video(self, video_file: str, output_folder: str) -> None:
        """Perform gaze prediction over a video and for each person"""

        logger.info(f"Video file: {video_file}")
        logger.info(f"Output folder: {output_folder}")
        logger.info(f"Device: {self.cfg.device}")

        # Video file
        check_file(video_file)

        # Output folder
        os.makedirs(output_folder, exist_ok=True)

        # Output files
        filename = os.path.splitext(os.path.basename(video_file))[0]
        heatmap_pid = self.cfg.demo.heatmap_pid
        hmap = f"-pid{heatmap_pid}" if heatmap_pid >= 0 else ""
        out_vid_file = os.path.join(output_folder, f"{filename}{hmap}-pred.mp4")
        out_pred_file = os.path.join(output_folder, f"{filename}-pred.csv")

        # Read video file
        cap = cv2.VideoCapture(video_file)
        ret, frame = cap.read()
        height, width = frame.shape[:2]
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize ffmpeg writer
        command = get_video_creation_command(out_vid_file, width, height, fps)
        process = sp.Popen(command, stdin=sp.PIPE)

        # Iterate over frames and process
        frame_nb = 0
        predictions = []
        with tqdm(total=frame_count) as pbar:
            while ret:
                frame_nb += 1

                # RGB image
                frame = frame[..., ::-1]  # BGR >> RGB

                # Run network to get gaze predictions
                pred = self.predict(frame)

                # Forward video
                num_people = len(pred["head_bboxes"])
                if num_people == 0:  # Not detection
                    process.stdin.write(frame.tobytes())
                    ret, frame = cap.read()
                    pbar.update(1)
                    continue

                # Get social gaze predictions (post-processing)
                social_gaze_preds = get_social_gaze_predictions(
                    pred, width, height, num_people
                )

                # Update prediction
                pred["lah"] = social_gaze_preds[0]
                pred["laeo"] = social_gaze_preds[1]
                pred["coatt"] = social_gaze_preds[2]
                pred["frame_nb"] = frame_nb
                predictions.append(pred)

                # Draw gaze predictions
                frame = draw_gaze(
                    frame,
                    social_gaze_preds,
                    head_bboxes=pred["head_bboxes"],
                    gaze_points=pred["gaze_points"],
                    gaze_vecs=pred["gaze_vecs"]
                    if self.cfg.visualization.show_gaze_vec
                    else None,
                    inouts=pred["inouts"],
                    pids=pred["pids"],
                    gaze_heatmaps=pred["gaze_heatmaps"],
                    heatmap_pid=heatmap_pid if heatmap_pid >= 0 else None,
                    frame_nb=frame_nb if self.cfg.visualization.show_frame_nb else None,
                    alpha=self.cfg.demo.alpha,
                    gaze_pt_size=self.cfg.visualization.gaze_point_size,
                    head_center_size=self.cfg.visualization.head_center_size,
                    thickness=self.cfg.visualization.thickness,
                    font_scale=self.cfg.visualization.font_scale,
                    colors=list(self.cfg.visualization.colors),
                    io_thr=self.cfg.demo.inout_thr,
                )

                # Write frame
                process.stdin.write(frame.tobytes())

                # Read next frame
                ret, frame = cap.read()
                pbar.update(1)

        # Save predictions
        self.save_predictions(predictions, out_pred_file, width, height)

        # Reinitialize tracker
        self.head_tracker.init_tracker()

        # Release Capture Device
        cap.release()

        # Close and flush stdin
        process.stdin.close()

        # Wait for sub-process to finish
        process.wait()

        # Terminate the sub-process
        process.terminate()
