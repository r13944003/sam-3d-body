# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import os
from glob import glob

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample_together
from tqdm import tqdm
import pickle


def main(args):
    base_output_folder = args.output_folder

    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=mhr_path
    )

    human_detector, human_segmentor, fov_estimator = None, None, None

    if args.detector_name:
        from tools.build_detector import HumanDetector
        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )

    if (args.segmentor_name == "sam2" and len(segmentor_path)) or args.segmentor_name != "sam2":
        from tools.build_sam import HumanSegmentor
        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )

    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator
        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    image_extensions = [
        "*.jpg", "*.jpeg", "*.png", "*.gif",
        "*.bmp", "*.tiff", "*.webp",
    ]

    subfolders = sorted(
        [p for p in glob(os.path.join(args.image_folder, "*")) if os.path.isdir(p)]
    )

    for subfolder in subfolders:
        images_dir = os.path.join(subfolder, "images")
        if not os.path.isdir(images_dir):
            continue

        images_list = sorted(
            [
                image
                for ext in image_extensions
                for image in glob(os.path.join(images_dir, ext))
            ]
        )
        if len(images_list) == 0:
            continue

        subfolder_name = os.path.basename(subfolder)

        if base_output_folder == "":
            output_folder = os.path.join("./output", subfolder_name)
        else:
            output_folder = os.path.join(base_output_folder, subfolder_name)

        os.makedirs(output_folder, exist_ok=True)

        for image_path in tqdm(images_list, desc=subfolder_name):
            base = os.path.splitext(os.path.basename(image_path))[0]
            out_pkl = os.path.join(output_folder, f"{base}.pkl")
            out_img = os.path.join(output_folder, f"{base}.jpg")
            out_no_det = os.path.join(output_folder, f"{base}_no_det.jpg")

            # =========================
            # Resume: if pkl exists → skip
            # =========================
            if os.path.isfile(out_pkl):
                continue

            outputs = estimator.process_one_image(
                image_path,
                bbox_thr=args.bbox_thresh,
                use_mask=args.use_mask,
            )

            # 永遠存 pkl（包含空結果）
            with open(out_pkl, "wb") as f:
                pickle.dump(
                    {
                        "image_path": image_path,
                        "outputs": outputs,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            # 如果沒有偵測到
            if outputs is None or len(outputs) == 0:
                if not args.no_save:
                    img = cv2.imread(image_path)
                    if img is not None:
                        cv2.imwrite(out_no_det, img)
                continue

            # 如果完全不需要視覺化
            if args.no_vis:
                continue

            img = cv2.imread(image_path)
            if img is None:
                continue

            rend_img = visualize_sample_together(
                img, outputs, estimator.faces
            )

            if not args.no_save:
                cv2.imwrite(out_img, rend_img.astype(np.uint8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - Single Image Human Mesh Recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python demo.py --image_folder ./images --checkpoint_path ./checkpoints/model.ckpt

                Environment Variables:
                SAM3D_MHR_PATH: Path to MHR asset
                SAM3D_DETECTOR_PATH: Path to human detection model folder
                SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
                SAM3D_FOV_PATH: Path to fov estimation model folder
                """,
    )

    parser.add_argument("--image_folder", required=True, type=str,
                        help="Path to folder containing input images")
    parser.add_argument("--output_folder", default="", type=str,
                        help="Path to output folder (default: ./output/<image_folder_name>)")
    parser.add_argument("--checkpoint_path",
                        default="./checkpoints/sam-3d-body-dinov3/model.ckpt",
                        type=str,
                        help="Path to SAM 3D Body model checkpoint")
    parser.add_argument("--detector_name", default="vitdet", type=str,
                        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).")
    parser.add_argument("--segmentor_name", default="sam2", type=str,
                        help="Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).")
    parser.add_argument("--fov_name", default="moge2", type=str,
                        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).")
    parser.add_argument("--detector_path", default="", type=str,
                        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)")
    parser.add_argument("--segmentor_path", default="", type=str,
                        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)")
    parser.add_argument("--fov_path", default="", type=str,
                        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)")
    parser.add_argument("--mhr_path",
                        default="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
                        type=str,
                        help="Path to MoHR/assets folder (or set SAM3D_mhr_path)")
    parser.add_argument("--bbox_thresh", default=0.8, type=float,
                        help="Bounding box detection threshold")
    parser.add_argument("--use_mask", action="store_true", default=False,
                        help="Use mask-conditioned prediction (segmentation mask is automatically generated from bbox)")

    # 新增的兩個 flag
    parser.add_argument("--no_vis", action="store_true", default=False,
                        help="Disable visualization")
    parser.add_argument("--no_save", action="store_true", default=False,
                        help="Disable saving images")

    args = parser.parse_args()
    main(args)