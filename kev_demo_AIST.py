# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import os
from glob import glob
from concurrent.futures import ThreadPoolExecutor
import time
import pickle

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


def _imread_bgr(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cv2.imread failed: {path}")
    return img


def main(args):
    base_output_folder = args.output_folder  # may be ""

    # env fallback
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # -------- perf knobs (GPU) --------
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    amp_dtype = None
    if args.amp == "fp16":
        amp_dtype = torch.float16
    elif args.amp == "bf16":
        amp_dtype = torch.bfloat16
    elif args.amp == "none":
        amp_dtype = None
    else:
        raise ValueError(f"Unknown amp: {args.amp}")

    # -------- load models once --------
    model, model_cfg = load_sam_3d_body(args.checkpoint_path, device=device, mhr_path=mhr_path)

    human_detector, human_segmentor, fov_estimator = None, None, None
    if args.detector_name:
        from tools.build_detector import HumanDetector
        human_detector = HumanDetector(name=args.detector_name, device=device, path=detector_path)

    if (args.segmentor_name == "sam2" and len(segmentor_path)) or args.segmentor_name != "sam2":
        from tools.build_sam import HumanSegmentor
        human_segmentor = HumanSegmentor(name=args.segmentor_name, device=device, path=segmentor_path)

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

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]

    # iterate direct subfolders under args.image_folder
    subfolders = sorted([p for p in glob(os.path.join(args.image_folder, "*")) if os.path.isdir(p)])

    # async saver
    saver = ThreadPoolExecutor(max_workers=args.save_workers) if args.async_save else None

    def submit_imwrite(path: str, img: np.ndarray):
        if args.no_save:
            return
        if saver is None:
            cv2.imwrite(path, img)
        else:
            saver.submit(cv2.imwrite, path, img)

    def write_done_marker(folder: str):
        if args.write_done:
            with open(os.path.join(folder, "_DONE"), "w", encoding="utf-8") as f:
                f.write("done\n")

    total_start = time.time()
    processed_images = 0
    skipped_images = 0

    with torch.inference_mode():
        for subfolder in subfolders:
            images_dir = os.path.join(subfolder, "images")
            if not os.path.isdir(images_dir):
                continue

            subfolder_name = os.path.basename(subfolder)
            if base_output_folder == "":
                output_folder = os.path.join("./output", subfolder_name)
            else:
                output_folder = os.path.join(base_output_folder, subfolder_name)
            os.makedirs(output_folder, exist_ok=True)

            # folder-level resume
            done_path = os.path.join(output_folder, "_DONE")
            if args.resume_folder and os.path.isfile(done_path):
                if args.verbose:
                    print(f"[SKIP FOLDER] {subfolder_name} (found _DONE)")
                continue

            images_list = sorted([img for ext in image_extensions for img in glob(os.path.join(images_dir, ext))])
            if args.limit > 0:
                images_list = images_list[: args.limit]
            if len(images_list) == 0:
                continue

            pbar = tqdm(images_list, desc=f"{subfolder_name}", leave=True)

            for image_path in pbar:
                base = os.path.splitext(os.path.basename(image_path))[0]
                out_pkl = os.path.join(output_folder, f"{base}.pkl")
                out_jpg = os.path.join(output_folder, f"{base}.jpg")
                out_no_det = os.path.join(output_folder, f"{base}_no_det.jpg")

                # image-level resume: if pkl exists, consider processed
                if args.resume_image and os.path.isfile(out_pkl):
                    skipped_images += 1
                    continue

                # read image once (for visualization / debug saving)
                img = None
                if (not args.no_vis) or (not args.no_save):
                    try:
                        img = _imread_bgr(image_path)
                    except Exception:
                        # even if read fails, still write pkl with error info
                        with open(out_pkl, "wb") as f:
                            pickle.dump({"image_path": image_path, "outputs": None, "error": "imread_failed"}, f,
                                        protocol=pickle.HIGHEST_PROTOCOL)
                        processed_images += 1
                        continue

                # inference (try passing ndarray if estimator supports; else fallback to path)
                if amp_dtype is not None and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        try:
                            outputs = estimator.process_one_image(img, bbox_thr=args.bbox_thresh, use_mask=args.use_mask)
                        except Exception:
                            outputs = estimator.process_one_image(image_path, bbox_thr=args.bbox_thresh, use_mask=args.use_mask)
                else:
                    try:
                        outputs = estimator.process_one_image(img, bbox_thr=args.bbox_thresh, use_mask=args.use_mask)
                    except Exception:
                        outputs = estimator.process_one_image(image_path, bbox_thr=args.bbox_thresh, use_mask=args.use_mask)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                # always save pkl (even empty outputs)
                with open(out_pkl, "wb") as f:
                    pickle.dump({"image_path": image_path, "outputs": outputs}, f, protocol=pickle.HIGHEST_PROTOCOL)

                # no detection -> optional debug save
                if outputs is None or len(outputs) == 0:
                    if (not args.no_save) and (img is not None) and args.save_no_det:
                        submit_imwrite(out_no_det, img)
                    processed_images += 1
                    continue

                # visualization (CPU heavy) + save
                if not args.no_vis:
                    rend_img = visualize_sample_together(img, outputs, estimator.faces).astype(np.uint8)
                    if not args.no_save:
                        submit_imwrite(out_jpg, rend_img)

                processed_images += 1

            # mark folder done
            write_done_marker(output_folder)

    if saver is not None:
        saver.shutdown(wait=True)

    total_time = time.time() - total_start
    print("\n=== Summary ===")
    print(f"processed_images: {processed_images}")
    print(f"skipped_images:   {skipped_images}")
    print(f"total_time:       {total_time:.2f}s")
    if processed_images > 0:
        print(f"throughput:       {processed_images/total_time:.2f} img/s (processed only)")
    print("===============\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - GPU-optimized + Resume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image_folder", required=True, type=str, help="Path to folder containing input subfolders")
    parser.add_argument("--output_folder", default="", type=str, help="Base output folder (default ./output/<subfolder>)")
    parser.add_argument("--checkpoint_path", default="./checkpoints/sam-3d-body-dinov3/model.ckpt", type=str)

    parser.add_argument("--detector_name", default="vitdet", type=str)
    parser.add_argument("--segmentor_name", default="sam2", type=str)
    parser.add_argument("--fov_name", default="moge2", type=str)

    parser.add_argument("--detector_path", default="", type=str)
    parser.add_argument("--segmentor_path", default="", type=str)
    parser.add_argument("--fov_path", default="", type=str)
    parser.add_argument("--mhr_path", default="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt", type=str)

    parser.add_argument("--bbox_thresh", default=0.8, type=float)
    parser.add_argument("--use_mask", action="store_true", default=False)

    # perf / workflow
    parser.add_argument("--amp", default="fp16", choices=["none", "fp16", "bf16"], help="Use Tensor Cores on 3090/4090")
    parser.add_argument("--no_vis", action="store_true", help="Disable visualization (CPU heavy)")
    parser.add_argument("--no_save", action="store_true", help="Disable saving images (I/O heavy)")
    parser.add_argument("--async_save", action="store_true", help="Save images with background threads")
    parser.add_argument("--save_workers", type=int, default=4)

    # resume
    parser.add_argument("--resume_folder", action="store_true", help="Skip subfolder if output has _DONE")
    parser.add_argument("--resume_image", action="store_true", help="Skip image if <base>.pkl exists")
    parser.add_argument("--write_done", action="store_true", help="Write _DONE after finishing each subfolder")
    parser.add_argument("--save_no_det", action="store_true", help="Save <base>_no_det.jpg for empty outputs")

    parser.add_argument("--limit", type=int, default=0, help="Limit images per subfolder for quick tests")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    main(args)