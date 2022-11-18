import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import convert_rgb_to_y


def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_patches = []
    hr_patches = []
    ###尺度增强策略
    for scale in [2]:
        for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
            hr = pil_image.open(image_path).convert('RGB')
            hr_width = (hr.width // scale) * scale
            hr_height = (hr.height // scale) * scale
            hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
            lr = hr.resize((hr_width // scale, hr_height // scale), resample=pil_image.BICUBIC)
            lr = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)
            hr = convert_rgb_to_y(hr)
            lr = convert_rgb_to_y(lr)

            for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
                for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):

                    lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                    hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    print('训练集的数量是',len(lr_patches))
    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()



def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')
    count=0
    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)
        count+=1
        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()
    print('测试集的数量',count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, default='./data/test/Set5')
    parser.add_argument('--output-path', type=str, default='./set5.h5')
    parser.add_argument('--patch-size', type=int, default=33)
    parser.add_argument('--stride', type=int, default=33)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--eval', type=bool,default=True)
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)
