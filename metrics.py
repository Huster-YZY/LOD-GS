#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr, psnr_mask
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir, masks_dir, res_level = None):
    renders = []
    gts = []
    masks = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        if res_level is not None:
            img_num = int(fname[:-4])
            if img_num % 4 != res_level:
                continue
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        mask = Image.open(masks_dir/fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        masks.append(tf.to_tensor(mask).unsqueeze(0).cuda())
        image_names.append(fname)
    return renders, gts, masks, image_names

def evaluate_multires(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            resolutions = [1,2,4,8]
            for res in range(4):
                for method in os.listdir(test_dir):
                    print("Method:", method)

                    full_dict[scene_dir][method] = {}
                    per_view_dict[scene_dir][method] = {}
                    full_dict_polytopeonly[scene_dir][method] = {}
                    per_view_dict_polytopeonly[scene_dir][method] = {}

                    method_dir = test_dir / method
                    gt_dir = method_dir/ "gt"
                    renders_dir = method_dir / "renders"
                    masks_dir = method_dir / "masks"
                    renders, gts, masks, image_names = readImages(renders_dir, gt_dir, masks_dir, res)

                    ssims = []
                    psnrs = []
                    lpipss = []
                    psnr_masks = []

                    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                        ssims.append(ssim(renders[idx], gts[idx]))
                        psnrs.append(psnr(renders[idx], gts[idx]))
                        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                        psnr_masks.append(psnr_mask(renders[idx],gts[idx],masks[idx]))

                    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                    print("  Masked PSNR: {:>12.7f}".format(torch.tensor(psnr_masks).mean(), ".5"))
                    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                    print("")

                    full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                            "PSNR": torch.tensor(psnrs).mean().item(),
                                                            "Masked PSNR": torch.tensor(psnr_masks).mean().item(),
                                                            "LPIPS": torch.tensor(lpipss).mean().item()})
                    per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                                "Masked PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnr_masks).tolist(), image_names)},
                                                                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

                with open(scene_dir + f"/results_{resolutions[res]}.json", 'w') as fp:
                    json.dump(full_dict[scene_dir], fp, indent=True)
                with open(scene_dir + f"/per_view_{resolutions[res]}.json", 'w') as fp:
                    json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                masks_dir = method_dir / "masks"
                renders, gts, masks, image_names = readImages(renders_dir, gt_dir, masks_dir)

                ssims = []
                psnrs = []
                lpipss = []
                psnr_masks = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    psnr_masks.append(psnr_mask(renders[idx],gts[idx],masks[idx]))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  Masked PSNR: {:>12.7f}".format(torch.tensor(psnr_masks).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "Masked PSNR": torch.tensor(psnr_masks).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "Masked PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnr_masks).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

def evaluate_res(model_paths, mid_point, mid_p2):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                masks_dir = method_dir / "masks"
                renders, gts, masks, image_names = readImages(renders_dir, gt_dir, masks_dir)

                ssims = []
                psnrs = []
                lpipss = []
                psnr_masks = []

                for idx in tqdm(range(mid_point), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    psnr_masks.append(psnr_mask(renders[idx],gts[idx],masks[idx]))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  Masked PSNR: {:>12.7f}".format(torch.tensor(psnr_masks).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "Masked PSNR": torch.tensor(psnr_masks).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "Masked PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnr_masks).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

                with open(scene_dir + "/results_l2.json", 'w') as fp:
                    json.dump(full_dict[scene_dir], fp, indent=True)
                with open(scene_dir + "/per_view_l2.json", 'w') as fp:
                    json.dump(per_view_dict[scene_dir], fp, indent=True)


                ssims = []
                psnrs = []
                lpipss = []
                psnr_masks = []

                for idx in tqdm(range(mid_point, mid_p2), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    psnr_masks.append(psnr_mask(renders[idx],gts[idx],masks[idx]))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  Masked PSNR: {:>12.7f}".format(torch.tensor(psnr_masks).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "Masked PSNR": torch.tensor(psnr_masks).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names[mid_point: mid_p2])},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names[mid_point: mid_p2])},
                                                            "Masked PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnr_masks).tolist(), image_names[mid_point: mid_p2])},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names[mid_point: mid_p2])}})

                with open(scene_dir + "/results_l3.json", 'w') as fp:
                    json.dump(full_dict[scene_dir], fp, indent=True)
                with open(scene_dir + "/per_view_l3.json", 'w') as fp:
                    json.dump(per_view_dict[scene_dir], fp, indent=True)
                
                ssims = []
                psnrs = []
                lpipss = []
                psnr_masks = []

                for idx in tqdm(range(mid_p2, len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    psnr_masks.append(psnr_mask(renders[idx],gts[idx],masks[idx]))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  Masked PSNR: {:>12.7f}".format(torch.tensor(psnr_masks).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "Masked PSNR": torch.tensor(psnr_masks).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names[mid_p2: len(renders)])},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names[mid_p2: len(renders)])},
                                                            "Masked PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnr_masks).tolist(), image_names[mid_p2: len(renders)])},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names[mid_p2: len(renders)])}})

                with open(scene_dir + "/results_l1.json", 'w') as fp:
                    json.dump(full_dict[scene_dir], fp, indent=True)
                with open(scene_dir + "/per_view_l1.json", 'w') as fp:
                    json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    # evaluate(args.model_paths)
    # evaluate_res(args.model_paths, 200, 220)
    evaluate_multires(args.model_paths)
