import argparse
import os
import copy
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from models import Net
from srcnn import SRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr,ssim,save_checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default='./train.h5')
    parser.add_argument('--eval-file', type=str, default='./set5.h5')
    parser.add_argument('--outputs-dir', type=str, default='./network')
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument("--resume", default="./network/models.pth", type=str, help="Path to checkpoint, Default=None")
    parser.add_argument("--pretrained", default="", type=str, help='path to pretrained models, Default=None')
    parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument("--step", type=int, default=2000,
                        help="Sets the learning rate to the initial LR decayed by momentum every n epochs")
    args = parser.parse_args()
    save_data={
    }
    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device',device)

    torch.manual_seed(args.seed)

    model = Net().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=args.lr)

    if os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume,map_location=lambda storage,loc:storage)
        args.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["models"].state_dict())
        optimizer.load_state_dict(checkpoint['optimizer'])
        save_data=torch.load('./fig/save_data.pth')
        args.lr=checkpoint['lr']
        best_epoch=checkpoint['best_epoch']
        best_psnr=checkpoint['best_psnr']
        print("===> loading checkpoint: {},start_epoch: {} ".format(args.resume,args.start_epoch))
    else:
        print("===> no checkpoint found at {}".format(args.resume))
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("===> load models {}".format(args.pretrained))
            weights = torch.load(args.pretrained,map_location=lambda storage,loc:storage)
            model.load_state_dict(weights['models'].state_dict())
        else:
            print("===> no models found at {}".format(args.pretrained))

    train_dataset = TrainDataset(args.train_file)
    lenth = 100
#     train_dataset,_=torch.utils.data.random_split(train_dataset,[lenth,len(train_dataset)-lenth])
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    print('导入数据集成功，训练集数量{}、测试集数量{}'.format(len(train_dataset), len(eval_dataset)))
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.start_epoch, args.num_epochs+1):
        lr = args.lr * (0.25 ** ((epoch) // args.step))
        print('epoch:', epoch, 'lr:', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        model.train()
        train_loss = AverageMeter()
        test_loss = AverageMeter()
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs=data['lr'].to(device)
                labels=data['hr'].to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                train_loss.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(train_loss.avg))
                t.update(len(inputs))

#         torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        test_psnr = AverageMeter()
        test_ssim = AverageMeter()

        for data in eval_dataloader:
            inputs = data['lr'].to(device)
            labels = data['hr'].to(device)
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)
                loss = criterion(preds, labels)
            test_loss.update(loss.item(), len(inputs))
            test_ssim.update(ssim(preds, labels), len(inputs))
            test_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f},eval ssim:{:.2f} eval loss :{}'.format(test_psnr.avg, test_ssim.avg, test_loss.avg))
        save_data[epoch] = {'train_loss': train_loss.avg, 'test_loss': test_loss.avg, 'test_psnr': test_psnr.avg,
                            'test_ssim': test_ssim.avg}
        torch.save(save_data, './fig/save_data.pth')

        if test_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = test_psnr.avg
            best_weights = copy.deepcopy(model)
            print('best_epoch:',best_epoch)
            ###保存整个网络
            torch.save(best_weights, os.path.join('./network', 'best.pth'))  ###直接保存整个模型
        save_checkpoint('./network', model, epoch,  optimizer,lr,best_epoch,best_psnr)
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))

