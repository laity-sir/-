from skimage.measure import compare_ssim, compare_psnr, compare_mse
import cv2,torch
from torchvision import transforms
import numpy as np
img1 = cv2.imread('./baby__bicubic2.png')[...,0]
img2 = cv2.imread('./baby__hr2.png')[...,0]

def psnr(img1,img2,data_range=255):
    if torch.is_tensor(img1):
        if img1.dim()==4:
            img1=img1.squeeze(0)
            img2=img2.squeeze(0)
        img1=img1.permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
        img2=img2.permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
    out=compare_psnr(img1,img2,data_range=data_range)
    return out
def ssim(img1,img2):
    if torch.is_tensor(img1):
        if img1.dim() == 4:
            img1 = img1.squeeze(0)
            img2 = img2.squeeze(0)
        img1 = img1.permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
        img2 = img2.permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
    out=compare_ssim(img1,img2,multichannel=True)
    return out

psnr1 = compare_psnr(img1, img2,data_range=255)
ssim1 = compare_ssim(img1, img2, multichannel=False)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
mse = compare_mse(img1, img2)

print('PSNR：{}，SSIM：{}，MSE：{}'.format(psnr1, ssim1, mse))

img1=transforms.ToTensor()(img1)
img2=transforms.ToTensor()(img2)

psnr2=psnr(img1,img2)
ssim2=ssim(img1,img2)
print('PSNR：{}，SSIM：{}，MSE：{}'.format(psnr2, ssim2, mse))