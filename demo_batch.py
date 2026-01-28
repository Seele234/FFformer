## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

##--------------------------------------------------------------
##------- Batch processing script for Restormer ---------
## 专门用于批量处理 fgq_jd/train/images 中 1.jpg 到 220.jpg 的图片
## 用法: python demo_batch.py --task Exposure --input_dir '/nas_nfs/Restormer-main/Exposure/fgq_jd/train/images' --result_dir '/nas_nfs/Restormer-main/Exposure/fgq_jd/train/restored'
##--------------------------------------------------------------

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
from pdb import set_trace as stx
import numpy as np

parser = argparse.ArgumentParser(description='Batch process Restormer on specific range of images')
parser.add_argument('--input_dir', default='/nas_nfs/Restormer-main/Exposure/fgq_jd/train/images', type=str,
                    help='Directory of input images')
parser.add_argument('--result_dir', default='/nas_nfs/Restormer-main/Exposure/fgq_jd/train/restored', type=str,
                    help='Directory for restored results')
parser.add_argument('--task', required=True, type=str, help='Task to run', choices=['Motion_Deblurring',
                                                                                    'Single_Image_Defocus_Deblurring',
                                                                                    'Deraining',
                                                                                    'Real_Denoising',
                                                                                    'FFformer',
                                                                                    'Exposure',
                                                                                    'Gaussian_Gray_Denoising',
                                                                                    'Gaussian_Color_Denoising'])
parser.add_argument('--start_idx', type=int, default=1, help='Start index of images to process')
parser.add_argument('--end_idx', type=int, default=220, help='End index of images to process')
parser.add_argument('--tile', type=int, default=None,
                    help='Tile size (e.g 720). None means testing on the original resolution image')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')

args = parser.parse_args()


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)


def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)


def get_weights_and_parameters(task, parameters):
    if task == 'Motion_Deblurring':
        weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
    elif task == 'Single_Image_Defocus_Deblurring':
        weights = os.path.join('Defocus_Deblurring', 'pretrained_models', 'single_image_defocus_deblurring.pth')
    elif task == 'Deraining':
        weights = os.path.join('Deraining', 'pretrained_models', 'deraining.pth')
    elif task == 'Real_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'real_denoising.pth')
        parameters['LayerNorm_type'] = 'BiasFree'
    elif task == 'FFformer':
        weights = os.path.join('Denoising', 'pretrained_models', 'FFformer_300000.pth')
        parameters['LayerNorm_type'] = 'BiasFree'
    elif task == 'Exposure':
        weights = os.path.join('Exposure', 'pretrained_models', 'exposure1.pth')
        parameters['LayerNorm_type'] = 'WithBias'
        parameters['use_enhanced_gfsa'] = False
        parameters['gaussian_kernel_size'] = 3
        parameters['shrinkage_ratio'] = 0.15
    elif task == 'Gaussian_Color_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'gaussian_color_denoising_blind.pth')
        parameters['LayerNorm_type'] = 'BiasFree'
    elif task == 'Gaussian_Gray_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'gaussian_gray_denoising_blind.pth')
        parameters['inp_channels'] = 1
        parameters['out_channels'] = 1
        parameters['LayerNorm_type'] = 'BiasFree'
    return weights, parameters


def get_specific_images(input_dir, start_idx, end_idx):
    """
    获取指定编号范围内的图片文件
    """
    files = []

    # 支持的图片格式
    extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']

    # 遍历所有支持格式的文件
    for ext in extensions:
        pattern = os.path.join(input_dir, f'*.{ext}')
        files.extend(glob(pattern))

    if len(files) == 0:
        raise Exception(f'No files found at {input_dir}')

    # 筛选指定编号的图片
    filtered_files = []
    for file_path in files:
        # 获取文件名（不含扩展名）
        filename = os.path.splitext(os.path.basename(file_path))[0]

        # 尝试将文件名转换为数字
        try:
            num = int(filename)
            if start_idx <= num <= end_idx:
                filtered_files.append(file_path)
        except ValueError:
            # 如果不是纯数字文件名，跳过
            continue

    # 按数字排序
    filtered_files = sorted(filtered_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    return filtered_files


task = args.task
inp_dir = args.input_dir
out_dir = os.path.join(args.result_dir, task)
start_idx = args.start_idx
end_idx = args.end_idx

os.makedirs(out_dir, exist_ok=True)

# 获取指定编号范围内的图片
files = get_specific_images(inp_dir, start_idx, end_idx)

if len(files) == 0:
    raise Exception(f'No files found in range {start_idx} to {end_idx} at {inp_dir}')

print(f"Found {len(files)} images to process: {files[:5]}...")  # 显示前5个文件

# Get model weights and parameters
parameters = {'inp_channels': 3,
              'out_channels': 3,
              'dim': 48,
              'num_blocks': [4, 6, 6, 8],
              'num_refinement_blocks': 4,
              'heads': [1, 2, 4, 8],
              'ffn_expansion_factor': 2.66,
              'bias': False,
              'LayerNorm_type': 'WithBias',
              'dual_pixel_task': False}

weights, parameters = get_weights_and_parameters(task, parameters)

# 任务分支
if task == 'FFformer' or task == 'Exposure':
    load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'FFformer_arch.py'))
    model = load_arch['FFformer'](**parameters)
else:
    load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
    model = load_arch['Restormer'](**parameters)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

checkpoint = torch.load(weights)
model.load_state_dict(checkpoint['params'])
model.eval()

img_multiple_of = 8

print(f"\n ==> Running {task} on images {start_idx} to {end_idx} with weights {weights}\n ")
print(f"Input directory: {inp_dir}")
print(f"Output directory: {out_dir}")
print(f"Number of images to process: {len(files)}\n")

with torch.no_grad():
    for idx, file_ in enumerate(tqdm(files, desc="Processing images")):
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        # 获取文件名和编号
        filename = os.path.splitext(os.path.basename(file_))[0]
        print(f"Processing image {idx + 1}/{len(files)}: {filename}.jpg")

        if task == 'Gaussian_Gray_Denoising':
            img = load_gray_img(file_)
        else:
            img = load_img(file_)

        input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)

        # Pad the input if not_multiple_of 8
        height, width = input_.shape[2], input_.shape[3]
        H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (width + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        if args.tile is None:
            ## Testing on the original resolution image
            restored = model(input_)
        else:
            # test the image tile by tile
            b, c, h, w = input_.shape
            tile = min(args.tile, h, w)
            assert tile % 8 == 0, "tile size should be multiple of 8"
            tile_overlap = args.tile_overlap

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            E = torch.zeros(b, c, h, w).type_as(input_)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = input_[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                    W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
            restored = E.div_(W)

        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:, :, :height, :width]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        if task == 'Gaussian_Gray_Denoising':
            save_gray_img((os.path.join(out_dir, f'{filename}.png')), restored)
        else:
            save_img((os.path.join(out_dir, f'{filename}.png')), restored)

    print(f"\nSuccessfully processed {len(files)} images")
    print(f"Restored images are saved at {out_dir}")
