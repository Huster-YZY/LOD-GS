import os
import sys
import json
import numpy as np
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--source_folder', type=str, default=None)
args = parser.parse_args(sys.argv[1:])
exp = "scales" #"levels"

if exp == "levels":
    scenes = ["chair", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
    if args.source_folder is None:
        print("Please input the source folder.")
        exit(-1)
    print(args.source_folder)
    data = {
        args.source_folder: ['l1_PSNR', 'l1_SSIM', 'l1_LPIPS', 'l2_PSNR', 'l2_SSIM', 'l2_LPIPS', 'l3_PSNR', 'l3_SSIM', 'l3_LPIPS']
    }

    for i in range(len(scenes)):
        scene = scenes[i]
        res_list = []
        for level in range(1,4):
            with open(os.path.join(args.source_folder, f"{scene}/results_l{level}.json"),"r") as f:
                res = json.load(f)
            res = res["ours_30000"]
            res_list += [res["PSNR"], res["SSIM"], res["LPIPS"]]
        data[scene] = res_list
        
    df = pd.DataFrame(data)
    excel_file = os.path.join(args.source_folder, 'output.xlsx')
    df.to_excel(excel_file, index=False, sheet_name='Sheet1')
    print(f"Data has been written to {excel_file}")

elif exp == "scales":
    scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
    if args.source_folder is None:
        print("Please input the source folder.")
        exit(-1)
    print(args.source_folder)
    data = {
        args.source_folder: ['l1_PSNR', 'l1_SSIM', 'l1_LPIPS', 'l2_PSNR', 'l2_SSIM', 'l2_LPIPS', 'l3_PSNR', 'l3_SSIM', 'l3_LPIPS', 'l4_PSNR', 'l4_SSIM', 'l4_LPIPS']
    }

    for i in range(len(scenes)):
        scene = scenes[i]
        res_list = []
        for scale in [1, 2, 4, 8]:
            with open(os.path.join(args.source_folder, f"{scene}/results_{scale}.json"),"r") as f:
                res = json.load(f)
            res = res["ours_30000"]
            res_list += [res["PSNR"], res["SSIM"], res["LPIPS"]]
        data[scene] = res_list
        
    df = pd.DataFrame(data)
    excel_file = os.path.join(args.source_folder, 'output.xlsx')
    df.to_excel(excel_file, index=False, sheet_name='Sheet1')
    print(f"Data has been written to {excel_file}")
