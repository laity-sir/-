import argparse
import os
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import  random_crop,convert_rgb_to_y
from PIL import Image

def train(args):
    if not os.path.exists("train"):
        os.makedirs("train")
    h5_file = h5py.File(args.output_path, 'w')
    lr_patches = []
    hr_patches = []
    flip = [0,1]
    count = 0
    if args.mode=='y':
        print('生成y通道数据')
    for image_path in sorted(os.listdir(args.images_dir)):
        hr = pil_image.open(os.path.join(args.images_dir,image_path)).convert('RGB')
        hr_images = []
        if args.mode=='y':
            # hr, _, _ = hr.convert('YCbCr').split()
            hr=np.array(hr)
            hr = convert_rgb_to_y(hr).astype(np.uint8)
            hr=Image.fromarray(hr)
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        if args.with_aug:
            for s in [1.0]:
                for r in [0,90,180,270]:
                    for c in range(len(flip)):
                        tmp = hr.resize((int(hr.width * s), int(hr.height * s)), resample=pil_image.BICUBIC)
                        if flip[c] == 1:  ##水平翻转
                            tmp = tmp.transpose(Image.FLIP_LEFT_RIGHT)
                        elif flip[c] == 2:  ###垂直翻转
                            tmp = tmp.transpose(Image.FLIP_TOP_BOTTOM)
                        else:
                            tmp = tmp  ##不翻转
                        tmp = tmp.rotate(r, expand=True)
                        hr_images.append(tmp)
        else:
            hr_images.append(hr)
        for hr in hr_images:
            for scale in [2]:
                args.scale = scale
                if args.samble_method=='m':
                    for i in range(0,hr_width-args.patch_size,args.patch_size):
                        for j in range(0,hr_height-args.patch_size,args.patch_size):
                            hr1 =  hr.crop((i, j, i + args.patch_size, j + args.patch_size))###随机裁剪
                            lr1 = hr1.resize((hr1.width // args.scale, hr1.height // args.scale),resample=pil_image.BICUBIC)
                            if args.bic:
                                lr1=lr1.resize((lr1.width*args.scale,lr1.height*args.scale),resample=pil_image.BICUBIC)
                            if args.save:
                                if not os.path.exists("./train/{}/lr/".format(args.scale)):
                                    os.makedirs("./train/{}/lr/".format(args.scale))
                                if not os.path.exists("./train/{}/hr/".format(args.scale)):
                                    os.makedirs("./train/{}/hr/".format(args.scale))
                                hr1.save(os.path.join("./train/{}/hr/".format(args.scale), '{}.png'.format(count)))
                                lr1.save(os.path.join("./train/{}/lr/".format(args.scale), '{}.png'.format(count)))
                            count += 1
                            hr1 = np.array(hr1).astype(np.float32)
                            lr1 = np.array(lr1).astype(np.float32)
                            lr_patches.append(lr1)
                            hr_patches.append(hr1)

                else:
                    for i in range(args.samble_num):
                        hr1=random_crop(hr,(args.patch_size,args.patch_size))
                        lr1 = hr1.resize((hr1.width // args.scale, hr1.height // args.scale), resample=pil_image.BICUBIC)
                        if args.bic:
                            lr1 = lr1.resize((hr1.width, hr1.height), resample=pil_image.BICUBIC)
                        count += 1
                        if args.save:
                            if not os.path.exists("./train/{}/lr/".format(args.scale)):
                                os.makedirs("./train/{}/lr/".format(args.scale))
                            if not os.path.exists("./train/{}/hr/".format(args.scale)):
                                os.makedirs("./train/{}/hr/".format(args.scale))
                            hr1.save(os.path.join("./train/{}/hr/".format(args.scale), '{}.png'.format(count)))
                            lr1.save(os.path.join("./train/{}/lr/".format(args.scale), '{}.png'.format(count)))

                        hr1 = np.array(hr1).astype(np.float32)
                        lr1 = np.array(lr1).astype(np.float32)
                        lr_patches.append(lr1)
                        hr_patches.append(hr1)
    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)
    h5_file.close()
    print('over,训练集数量{}'.format(count))

# def eval(args):
#     if not os.path.exists("test"):
#         os.makedirs("test")
#     h5_file = h5py.File(args.eval_output_path, 'w')
#     lr_group = h5_file.create_group('lr')
#     hr_group = h5_file.create_group('hr')
#     count=0
#     if args.mode=='y':
#         print('生成y通道数据')
#     else:
#         print('生成rgb数据')
#     args.scale=config.scale
#     for i, image_path in enumerate(sorted(os.listdir(args.eval_images_dir))):
#         hr = pil_image.open(os.path.join(args.eval_images_dir,image_path)).convert('RGB')
#         if args.mode=='y':  ###这里是生成y通道数据
#             hr,_,_=hr.convert('YCbCr').split()
#         hr_width = (hr.width // args.scale) * args.scale
#         hr_height = (hr.height // args.scale) * args.scale
#         hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
#         lr = hr.resize((hr.width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
#         if args.bic:
#             lr= lr.resize((lr.width*args.scale, lr.height*args.scale), resample=pil_image.BICUBIC) ###双三次插值
#         count += 1
#         if args.save:
#             if not os.path.exists("./test/{}/lr/".format(args.scale)):
#                 os.makedirs("./test/{}/lr/".format(args.scale))
#             if not os.path.exists("./test/{}/hr/".format(args.scale)):
#                 os.makedirs("./test/{}/hr/".format(args.scale))
#             hr.save(os.path.join("./test/{}/hr/".format(args.scale), '{}.png'.format(i)))
#             lr.save(os.path.join("./test/{}/lr/".format(args.scale), '{}.png'.format(i)))
#         hr = np.array(hr).astype(np.float32)
#         lr = np.array(lr).astype(np.float32)
#
#         lr_group.create_dataset(str(i), data=lr)
#         hr_group.create_dataset(str(i), data=hr)
#     h5_file.close()
#     print('测试集数量：{}'.format(count))

def eval(args):
    if not os.path.exists("test"):
        os.makedirs("test")
    for scale in [2,3,4]:
        if not os.path.exists('./test/{}'.format(scale)):
            os.makedirs('./test/{}'.format(scale))
        for dir_name in os.listdir(args.eval_images_dir):
            h5_file = h5py.File('./test/{}/{}.h5'.format(scale,dir_name), 'w')
            lr_group = h5_file.create_group('lr')
            hr_group = h5_file.create_group('hr')
            count=0
            if args.mode=='y':
                print('生成y通道数据')
            else:
                print('生成rgb数据')
            image_dir=os.path.join(args.eval_images_dir,dir_name)
            for i, image_path in enumerate(sorted(os.listdir(image_dir))):
                hr = pil_image.open(os.path.join(image_dir,image_path)).convert('RGB')
                # if args.mode=='y':  ###这里是生成y通道数据
                #     hr,_,_=hr.convert('YCbCr').split()
                hr_width = (hr.width // scale) * scale
                hr_height = (hr.height // scale) * scale
                hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
                lr = hr.resize((hr.width // scale, hr_height // scale), resample=pil_image.BICUBIC)
                if args.bic:
                    lr= lr.resize((lr.width*scale, lr.height*scale), resample=pil_image.BICUBIC) ###双三次插值
                count += 1
                if args.save:
                    if not os.path.exists("./test/{}/lr/".format(args.scale)):
                        os.makedirs("./test/{}/lr/".format(args.scale))
                    if not os.path.exists("./test/{}/hr/".format(args.scale)):
                        os.makedirs("./test/{}/hr/".format(args.scale))
                    hr.save(os.path.join("./test/{}/hr/".format(args.scale), '{}.png'.format(i)))
                    lr.save(os.path.join("./test/{}/lr/".format(args.scale), '{}.png'.format(i)))

                hr = np.array(hr).astype(np.float32)
                lr = np.array(lr).astype(np.float32)
                hr = convert_rgb_to_y(hr)
                lr = convert_rgb_to_y(lr)
                lr_group.create_dataset(str(i), data=lr)
                hr_group.create_dataset(str(i), data=hr)
            h5_file.close()
            print('dir_name:{} scale:{} 测试集数量：{}'.format(dir_name,scale,count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, default='./data/T91_HR')
    parser.add_argument('--output-path', type=str, default='./train/T91.h5')
    parser.add_argument('--eval_images-dir', type=str, default='./data/test')
    parser.add_argument('--eval-output-path', type=str, default='./test/set5.h5')
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--samble-method', type=str, default='m',help='m 表示全部采样，否则就是随机采样')
    parser.add_argument('--samble-num', type=int, default=2, help='随机采样的个数')
    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--save', type=bool, default=False, help='是否保存图片')
    parser.add_argument('--with-aug',type=bool,default=True,help='数据增强')
    parser.add_argument('--mode', type=str,help='y:表示ycbcr',default='y')
    parser.add_argument('--bic', type=bool, default=True)
    args = parser.parse_args()
    # train(args)
    eval(args)
