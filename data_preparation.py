import tensorflow as tf
import numpy as np
from models import MiniMaxCCA, CNNdecoder, CNNencoder, CNNDAE
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import scipy.io as sio
import glob
import PIL


class ViewDataset():
    def __init__(self, view1, view2):
        self.view1 = tf.Tensor(view1, dtype=tf.float32)
        self.view2 = tf.Tensor(view2, dtype=tf.float32)
        assert self.view1.shape[0] == self.view2.shape[0]
        self.len =  self.view1.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.view1[item], self.view2[item], item


def get_cars3d(filedir='./data/cars/'):
    tmp = []
    for f in glob.glob(filedir+'*mesh.mat'):
        a = sio.loadmat(f)
        tt = np.zeros((a['im'].shape[3], a['im'].shape[4], 64, 64, 3))
        for i in range(a['im'].shape[3]):
            for j in range(a['im'].shape[4]):
                pic = PIL.Image.fromarray(a['im'][:,:,:,i,j])
                pic.thumbnail((64, 64), PIL.Image.ANTIALIAS)
                tt[i, j, :, :, :] = np.array(pic)/255.

        tmp.append(np.array(tt))

    data = np.transpose(np.stack(tmp, 0))
    print(f'Data Shape: {data.shape}')
    dim0, dim1, dim2 = data.shape[:3]

    imgs = np.copy(data)
    print(f'Shape before: {imgs.shape}')
    imgs = np.transpose(imgs, (3, 5, 4, 0, 2, 1))
    print(f'Shape after: {imgs.shape}')

    # 4 elevations
    elv1 = imgs[0]
    elv2 = imgs[1]
    elv3 = imgs[2]
    elv4 = imgs[3]

    # Show samples
    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(np.transpose(elv1[0,1,:,:,:],(1,2,0)), interpolation='nearest')
    axarr[0].axis('off')
    axarr[1].imshow(np.transpose(elv2[0,7,:,:,:],(1,2,0)), interpolation='nearest')
    axarr[1].axis('off')
    axarr[2].imshow(np.transpose(elv3[0,14,:,:,:],(1,2,0)), interpolation='nearest')
    axarr[2].axis('off')
    axarr[3].imshow(np.transpose(elv4[0,21,:,:,:],(1,2,0)), interpolation='nearest')
    axarr[3].axis('off')

    fig.suptitle('Samples of both views, left two for view1, right two for view2',fontsize=10)
    plt.show()

    # Get the two views in order
    view1=[]
    view2=[]

    for i in range(elv1.shape[0]):
        # Lower elevations
        view1.append(elv1[i, :, :, :, :])
        view1.append(elv2[i, :, :, :, :])
        # Higher elevations
        view2.append(elv3[i, :, :, :, :])
        view2.append(elv4[i, :, :, :, :])

    view1 = np.concatenate(view1, axis=0)
    view2 = np.concatenate(view2, axis=0)

    return view1, view2


