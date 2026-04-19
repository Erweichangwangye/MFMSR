from __future__ import print_function
import lpips
import argparse
import cv2
import torch.nn.functional as F
import numpy as np
import torchvision
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform
from os import listdir
import math
# ---load model architecture---
# from model_archs.fremamba import FreMamba as net
# from model_archs.ms2d_mamba import MambaIR as net
from model_archs.mambairv2_arch import MambaIRv2 as net
# from model_archs.mambair_arch import MambaIR as net
import glob
import numpy as np
import socket
import time
from PIL import Image

# Test settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='training batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')

parser.add_argument('--data_dir', type=str, default='datasets/test/AID')

parser.add_argument('--model_type', type=str, default='mfmsr')
parser.add_argument('--pretrained_sr', default='/media/a204/742CCF0E2CCEC9F6/xym/FreMamba-main/saved_models/mfmsr/mambaIRV2_ALL/fmsr_epoch_200.pth', help='sr pretrained base model')
parser.add_argument('--save_folder', default='datasets/test_results/', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
cuda = opt.gpu_mode
print(opt)

current_time = time.strftime("%H-%M-%S")
opt.save_folder = opt.save_folder + current_time + '/'

if not os.path.exists(opt.save_folder):
    os.makedirs(opt.save_folder)

transform = transform.Compose([transform.ToTensor(),])
def PSNR(pred, gt):
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %f M' % (num_params / 1e6))

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # LPIPS 需要归一化到 [-1, 1]
    ])
    return transforms(img).unsqueeze(0).cuda()  # 添加 batch 维度并移至GPU

torch.cuda.manual_seed(opt.seed)
device = 'cuda:0'
print('===> Building model ', opt.model_type)
model = net()
model = torch.nn.DataParallel(model, device_ids=gpus_list)
print('---------- Networks architecture -------------')
print_network(model)
model = model.cuda(gpus_list[0])

model_name = os.path.join(opt.pretrained_sr)
if os.path.exists(model_name):
    # model= torch.load(model_name, map_location=lambda storage, loc: storage)
    model.load_state_dict(torch.load(model_name))
    print('Pre-trained SR model is loaded.')
else:
    print('No pre-trained model!!!!')

def eval(folder_name):
    print('===> Loading val datasets')
    LR_filename = os.path.join(opt.data_dir, 'LR') + '/' + folder_name
    GT_filename = os.path.join(opt.data_dir, 'GT') + '/' + folder_name
    LR_image = sorted(glob.glob(os.path.join(LR_filename, '*')))  # LR图像路径列表
    GT_image = sorted(glob.glob(os.path.join(GT_filename, '*')))  # LR图像路径列表
    # test begin
    model.eval()
    i=0.0
    total_psnr_lr=0
    total_ssim_lr=0
    # total_lpips_lr=0
    total_psnr_sr=0
    total_ssim_sr=0
    # total_lpips_sr=0


    total_psnr_lr_30=0
    total_ssim_lr_30=0
    # total_lpips_lr_30=0
    total_psnr_sr_30=0
    total_ssim_sr_30=0
    # total_lpips_sr_30=0
    for img_path, gt_path in zip(LR_image, GT_image):
        lr = Image.open(img_path).convert('RGB')
        lr = transform(lr).unsqueeze(0)
        with torch.no_grad():
            t0 = time.time()
            prediction = model(lr)
            t1 = time.time()

        prediction = prediction.cpu()
        prediction = prediction.data[0].numpy().astype(np.float32)
        prediction = prediction * 255.0
        prediction = prediction.clip(0, 255)
        prediction = prediction.transpose(1, 2, 0)

        print("===> Processing image: %s || Timer: %.4f sec." % (img_path, (t1 - t0)))
        save_name = os.path.splitext(os.path.basename(img_path))[0]
        save_foler = opt.save_folder + folder_name
        if not os.path.exists(save_foler):
            os.makedirs(save_foler)
        save_fn = save_foler + save_name + '.png'
        print('save image to:', save_fn)
        Image.fromarray(np.uint8(prediction)).save(save_fn)

        # loss_fn = lpips.LPIPS(net='alex', version='0.1').cuda()
        GTimage = cv2.imread(gt_path)  # 原图
        SRimage = cv2.imread(save_fn)  # 处理后的图
        LRimage = cv2.imread(img_path)  # 处理后的图
        Resize_LRimage=cv2.resize(LRimage,(GTimage.shape[1],GTimage.shape[0]))  # LR图

        ssim_lr = ssim(GTimage, Resize_LRimage, channel_axis=2) # ssim
        psnr_lr = psnr(GTimage, Resize_LRimage) # 计算 PSNR

        # gt = load_image(gt_path)
        # lrI = load_image(img_path)
        # gt_downsampled = F.interpolate(gt, size=lrI.shape[2:], mode='bicubic')
        # lpips_lr = loss_fn(gt_downsampled, lrI) #lpips

        total_psnr_lr = total_psnr_lr+psnr_lr
        total_ssim_lr = total_ssim_lr+ssim_lr
        # total_lpips_lr=total_lpips_lr+lpips_lr.item()
        total_psnr_lr_30 = total_psnr_lr_30+psnr_lr
        total_ssim_lr_30 = total_ssim_lr_30+ssim_lr
        # total_lpips_lr_30 = total_lpips_lr_30 + lpips_lr.item()
        with open(save_foler+'psnr_ssim.txt', 'a') as f:
            print(f'===> Processing image: {img_path}', file=f)
            print(f'SSIM LR: {ssim_lr:.4f}', file=f)
            print(f'PSNR LR: {psnr_lr:.2f} dB', file=f)
            # print(f'Lpips LR: {lpips_lr.item():.4f} dB', file=f)

        ssim_val = ssim(GTimage, SRimage, channel_axis=2) # ssim
        psnr_val = psnr(GTimage, SRimage) # 计算 PSNR
        # lpips_val = loss_fn(load_image(gt_path), load_image(save_fn))
        total_psnr_sr = total_psnr_sr+psnr_val
        total_ssim_sr = total_ssim_sr+ssim_val
        # total_lpips_sr=total_lpips_sr+lpips_val.item()
        total_psnr_sr_30 = total_psnr_sr_30+psnr_val
        total_ssim_sr_30 = total_ssim_sr_30+ssim_val
        # total_lpips_sr_30=total_lpips_sr_30+lpips_val.item()
        with open(save_foler+'psnr_ssim.txt', 'a') as f:
            print(f'SSIM SR: {ssim_val:.4f}', file=f)
            print(f'PSNR SR: {psnr_val:.2f} dB', file=f)
            # print(f'Lpips SR: {lpips_val.item():.4f} dB', file=f)

        i=i+1.0
        if i%30==0:
            with open(save_foler + 'psnr_ssim.txt', 'a') as f:
                print('===> Each 30 SSIM And PSNR', file=f)
                print(f'Total_Average_SSIM LR: {total_ssim_lr_30 / 30:.4f}', file=f)
                print(f'Total_Average_PSNR LR: {total_psnr_lr_30 / 30:.2f} dB', file=f)
                # print(f'Total_Average_PSNR LR: {total_lpips_lr_30 / 30:.2f} dB', file=f)
                print(f'Total_Average_SSIM SR: {total_ssim_sr_30 / 30:.4f}', file=f)
                print(f'Total_Average_PSNR SR: {total_psnr_sr_30 / 30:.2f} dB', file=f)
                # print(f'Total_Average_PSNR LR: {total_lpips_sr_30 / 30:.2f} dB', file=f)
                total_psnr_lr_30 = 0
                total_ssim_lr_30 = 0
                total_psnr_sr_30 = 0
                total_ssim_sr_30 = 0
    with open(save_foler + 'psnr_ssim.txt', 'a') as f:
        print('===> Average SSIM And PSNR', file=f)
        print(f'Total_Average_SSIM LR: {total_ssim_lr/i:.4f}', file=f)
        print(f'Total_Average_PSNR LR: {total_psnr_lr/i:.2f} dB', file=f)
        # print(f'Total_Average_PSNR LR: {total_lpips_lr/i:.2f} dB', file=f)
        print(f'Total_Average_SSIM SR: {total_ssim_sr/i:.4f}', file=f)
        print(f'Total_Average_PSNR SR: {total_psnr_sr/i:.2f} dB', file=f)
        # print(f'Total_Average_PSNR LR: {total_lpips_sr/i:.2f} dB', file=f)

if __name__ == '__main__':
    AID_class_name = ['Airport/','BareLand/','BaseballField/','Beach/','Bridge/','Center/','Church/','Commercial/','DenseResidential/',
                      'Desert/','Farmland/','Forest/','Industrial/','Meadow/','MediumResidential/','Mountain/','Park/','Parking/','Playground/',
                      'Pond/','Port/','RailwayStation/','Resort/','River/','School/','SparseResidential/','Square/','Stadium/','StorageTanks/','Viaduct/']
    dota_class = ['']
    for folder in dota_class:
        eval(folder_name=folder)
