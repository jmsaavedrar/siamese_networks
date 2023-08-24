import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
from io import BytesIO

def view(ds, n_rows) :    
    n_cols = 3
    _, ax = plt.subplots(n_rows, n_cols)
    
    ax[0,0].set_title('Anchor')
    ax[0,1].set_title('Positive')
    ax[0,2].set_title('Negative')
    for i in range(n_rows) :        
        ax[i,0].set_axis_off()
        ax[i,1].set_axis_off()
        ax[i,2].set_axis_off()
        
             
    for batch in ds :
        n = batch[0].shape[0]
        perm = np.random.permutation(n)
        perm = np.where(perm == np.arange(n), (perm + 1) % n, perm)
        #perm = (perm +  1) % n 
        for i in np.arange(n) :
            ax[i,0].imshow(batch[0][i]) #Anchor                                 
            ax[i,1].imshow(batch[1][i])
            ax[i,2].imshow(batch[0][perm[i]])
        plt.waitforbuttonpress(2)            
    plt.show()
# 
# 
def map_func(example_serialized):    
#     features_map=tfds.features.FeaturesDict({'image': tfds.features.Image(shape=(None, None, 3)),
#                                               'label': tfds.features.ClassLabel(names=range(100))})
#     features = tf.io.parse_example(example_serialized, features_map)
    image_anchor = example_serialized['image-anchor']
    image_positive = example_serialized['image_positive']
    print(image_anchor.shape, flush = True)    
    print(image_positive.shape, flush = True)
    image_anchor = tf.image.resize_with_pad(image_anchor, 256, 256)
    image_positive = tf.image.resize_with_pad(image_positive, 256, 256)
    image_anchor = tf.image.random_crop(image_anchor, size = [224, 224, 3])
    image_positive = tf.image.random_crop(image_positive, size = [224, 224, 3])
    image_positive = tf.cast(image_positive, tf.uint8)    
    image_anchor = tf.cast(image_anchor, tf.uint8)
    return image_anchor, image_positive 

if __name__ == '__main__' :
    #data_dir ='/mnt/hd-data/Datasets/imagenet/tfds'
    start = time.time()
    ds = tfds.load('tfds_look')
    end = time.time()
    et = int(((end - start )* 1000))
    print('ds loaded ok. Elapsed Time: {}'.format(et))
    
    batch_size = 5
    ds_train = ds['train'].map(map_func).shuffle(1024).batch(batch_size)        
    #ds_train = ds_train.take(10)
#     for sample in ds_test:
#         print(sample)    
    view(ds_train, 5)

