import argparse
import glob
import time
import torch
from torch.utils.data.dataloader import DataLoader
from datasets import  EvalDataset
from utils import AverageMeter, calc_psnr,ssim,psnr
import os
from models import Net

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-file', type=str, default='./test')
    parser.add_argument('--weights-file', type=str,default='./network/x2/best.pth')
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_dir_name=os.listdir(args.eval_file)
    print(test_dir_name)
    models = Net().to(device)
    print('loaded models')
    model=torch.load(args.weights_file, map_location=lambda storage, loc: storage).to(device)
    print('在y通道上进行测试')
    for i in range(len(test_dir_name)):
        print('正在测试的尺度因子',test_dir_name[i])
        hh=os.path.join(args.eval_file ,test_dir_name[i])
        for i in glob.glob('{}/*.h5'.format(hh)):
            print(i)
            eval_dataset=EvalDataset(i)
            eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
            model.eval()
            epoch_psnr = AverageMeter()
            epoch_ssim = AverageMeter()
            epoch_time=AverageMeter()
            bic_psnr=AverageMeter()
            bic_ssim=AverageMeter()
            end=time.time()
            for data in eval_dataloader:
                inputs = data['lr']
                labels=data['hr']
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    preds = model(inputs).clamp(0.0, 1.0)
                epoch_psnr.update(psnr(preds, labels), len(inputs))
                epoch_ssim.update(ssim(preds, labels),len(inputs))
                bic_psnr.update(psnr(inputs,labels),len(inputs))
                bic_ssim.update(ssim(inputs,labels),len(inputs))
            epoch_time.update(time.time()-end)
            print('eval-file:{}    eval psnr: {:.2f},eval ssim:{:2f}  epoch time:{:f}'.format(i, epoch_psnr.avg, epoch_ssim.avg,epoch_time.avg))
            print('bicubic and hr of psnr:{} ssim:{}'.format(bic_psnr.avg,bic_ssim.avg))