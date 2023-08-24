
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import configparser
import argparse

#---------------------------------------------------------------------------------------
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

        
AUTO = tf.data.AUTOTUNE
#---------------------------------------------------------------------------------------
import models.siamese as siamese

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type = str, required = True)    
    parser.add_argument('-model', type = str, choices = ['RESNET'], required = True)
    parser.add_argument('-gpu', type = int, required = False) # gpu = -1 set for using all gpus
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    config.read(args.config)
    config_model = config['RESNET']
    config_data = config['DATA']
    
    gpu_id = 0
    if not args.gpu is None :
        gpu_id = args.gpu
        
    ds = tfds.load('tfds_look')        
    ds_train = ds['train']    
    ds_train = (
        ds_train.shuffle(1024)
        .map(map_func, num_parallel_calls=AUTO)
        .batch(config_model.getint('BATCH_SIZE'))
        .prefetch(AUTO) )
    
    if gpu_id >= 0 :
        with tf.device('/device:GPU:{}'.format(gpu_id)) :
            model = siamese.Siamese(config_model, config_data)
            model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9),
                          loss= tf.keras.losses.CategoricalCrossentropy(),
                          metrics=['accuracy'])
                            
            model.fit(ds_train, epochs = config_model.getint('EPOCHS'))
        