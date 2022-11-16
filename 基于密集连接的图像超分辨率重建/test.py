import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import os,time
from models import Net
import matplotlib.pyplot as plt
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr,psnr,ssim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default='./network/x2/best.pth')
    parser.add_argument('--image-file', type=str, default='./data/test/Set5/baby.png')
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Net().to(device)

    model = torch.load(args.weights_file, map_location=lambda storage, loc: storage).to(device)

    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')
    basename = args.image_file.split('/')[-1]
    basename = basename.split('.')[0]  ###baby
    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    ##HR image
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    HR=image
    HR.save(os.path.join('./fig', basename + '__hr{}.png'.format(args.scale)))
    ##LR image
    image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
    ##bicubic image
    image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)

    image.save(os.path.join('./fig', basename + '__bicubic{}.png'.format(args.scale)))
    bicubic=image

    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        end = time.time()
        preds = model(y).clamp(0.0, 1.0)
        print('处理图片的时间', time.time() - end, 's')

    print('PSNR: {:.2f}'.format(calc_psnr(y, preds)))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(os.path.join('./fig', basename + '__pred{}.png'.format(args.scale)))

    print('bicubic and hr psnr:{}'.format(psnr(convert_rgb_to_ycbcr(np.array(HR))[...,0],convert_rgb_to_ycbcr(np.array(bicubic))[...,0])))
    print('pred and hr psnr:{}'.format(psnr(convert_rgb_to_ycbcr(np.array(HR))[...,0],convert_rgb_to_ycbcr(np.array(output))[...,0])))
    print('bicubic and hr ssim:{}'.format(ssim(convert_rgb_to_ycbcr(np.array(HR))[...,0],convert_rgb_to_ycbcr(np.array(bicubic))[...,0])))
    print('pred and hr ssim:{}'.format(ssim(convert_rgb_to_ycbcr(np.array(HR))[...,0],convert_rgb_to_ycbcr(np.array(output))[...,0])))

    ##显示
    plt.figure()
    plt.subplot(131)
    plt.imshow(HR)
    plt.xticks([])  # 去掉x轴的刻度
    plt.yticks([])
    plt.title("hr")
    plt.subplot(132)
    plt.imshow(bicubic)
    plt.xticks([])  # 去掉x轴的刻度
    plt.yticks([])
    plt.title("bicubic")
    plt.subplot(133)
    plt.imshow(output)
    plt.xticks([])  # 去掉x轴的刻度
    plt.yticks([])
    plt.title('pred')
    plt.show()




