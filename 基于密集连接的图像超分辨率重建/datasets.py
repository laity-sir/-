import h5py
from torch.utils.data import Dataset
import numpy as np

class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return {'lr':np.expand_dims(f['lr'][idx] / 255., 0), 'hr':np.expand_dims(f['hr'][idx] / 255., 0)}

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return {'lr':np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0),
                    'hr':np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)}

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

if __name__=='__main__':
    import matplotlib.pyplot as plt
    import random
    from PIL import Image
    import numpy as np
    from torchvision import transforms

    ###可视化训练集图片
    # num=40
    # col=8
    # row=int(num/8)
    # dataset=TrainDataset('./set5.h5')
    # print(dataset[0][1])
    # index = np.random.randint(1, len(dataset), num)
    # print(index)
    # hh=[dataset[i]['hr'] for i in index]
    # for i in range(num):
    #     for j in range(8):
    #         # plt.figure()
    #         plt.subplot(row, col, i+1)
    #         plt.xticks([])  # 去掉x轴的刻度
    #         plt.yticks([])  # 去掉y轴的刻度
    #         image=hh[i]*255
    #         image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    #         image=np.squeeze(image,0)
    #         image=Image.fromarray(image)
    #         plt.imshow(image, cmap='gray')
    # plt.show()
    ##可视化测试集图片

    dataset=EvalDataset('./test/2/Set5.h5')
    for i in range(len(dataset)):
        plt.xticks([])  # 去掉x轴的刻度
        plt.yticks([])  # 去掉y轴的刻度
        image=dataset[i]['hr']*255
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        image=np.squeeze(image,0)
        image=Image.fromarray(image)
        plt.imshow(image, cmap='gray')
        plt.show()

