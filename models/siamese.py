import tensorflow as tf
import configparser
import os
import matplotlib.pyplot as plt
import models.resnet as resnet

def view(anchors, positives, negatives):
    n = anchors.shape[0]
    _, ax = plt.subplots(n, 3)
    print(negatives.shape)        
    for i, image in  enumerate(anchors) :
        ax[i,0].imshow(image)
        ax[i,1].imshow(positives[i])  
        ax[i,2].imshow(negatives[i])
    plt.waitforbuttonpress(1)
        
class Siamese(tf.keras.Model):
    def __init__(self, config_model, config_data, **kwargs):        
        super(Siamese, self).__init__(**kwargs)        
        self.CROP_SIZE = config_data.getint('CROP_SIZE')
        self.PROJECT_DIM =  config_model.getint('PROJECT_DIM')
        self.WEIGHT_DECAY = config_model.getfloat('WEIGHT_DECAY')            
        self.MARGIN = config_model.getfloat('MARGIN')
        self.CHANNELS = 3        
        self.encoder = self.get_encoder()
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.dist_pos_tracker = tf.keras.metrics.Mean(name="dist_pos")
        self.dist_neg_tracker = tf.keras.metrics.Mean(name="dist_neg")
        
    def get_encoder(self):       
        inputs = tf.keras.layers.Input((self.CROP_SIZE, self.CROP_SIZE, self.CHANNELS))                
        x = inputs # / 127.5 - 1
        #the backbone can be an input to the clas SimSiam
        bkbone = resnet.ResNetBackbone([3,4,6,3], [64,128, 256, 512], kernel_regularizer = tf.keras.regularizers.l2(self.WEIGHT_DECAY))        
        x = bkbone(x)   
        # Projection head.
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.PROJECT_DIM)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(self.PROJECT_DIM)(x)

#         #x = tf.keras.layers.Flatten()(x)        
#         x = tf.keras.layers.Dense(
#             self.PROJECT_DIM, 
#             use_bias=True, 
#             kernel_regularizer=tf.keras.regularizers.l2(self.WEIGHT_DECAY)
#         )(x)
        #outputs = tf.keras.layers.BatchNormalization()(x)
        
        outputs = tf.math.l2_normalize(x, axis=1)
        
        return tf.keras.Model(inputs, outputs, name="encoder")
     
    def compute_loss(self, xa, xp, xn):                            
        margin = self.MARGIN
        dist_pos = tf.sqrt(tf.reduce_sum(tf.math.square(xa - xp), axis = 1))
        dist_neg = tf.sqrt(tf.reduce_sum(tf.math.square(xa - xn), axis = 1))
        #dist_pos  = tf.math.sqrt(2.0 - 2.0*tf.reduce_sum((xa * xp), axis = 1))
        #dist_neg  = tf.math.sqrt(2.0 - 2.0*tf.reduce_sum((xa * xn), axis = 1))
        loss = tf.math.maximum(0.0, dist_pos - dist_neg + margin)
                        
        return tf.reduce_mean(loss), tf.reduce_mean(dist_pos), tf.reduce_mean(dist_neg)
                
                                    
    def train_step(self, batch):
        # Unpack the data.
        anchors, positives = batch
        # select negatives
        n = tf.shape(anchors)[0]
        pos = tf.range(n)
        perm = tf.random.shuffle(pos)
        perm = tf.where(perm == pos, (perm + 1) % n, perm)
        negatives = tf.gather(anchors, perm)
        #view(anchors, positives, negatives)
        # training one step
        with tf.GradientTape() as tape:
            xa = self.encoder(anchors)
            xp = self.encoder(positives)
            xn = self.encoder(negatives)            
            loss, dist_pos, dist_neg = self.compute_loss(xa, xp, xn)
        
        # Compute gradients and update the parameters.
        learnable_params = (
            self.encoder.trainable_variables
        )
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # tracking status.        
        self.loss_tracker.update_state(loss)
        self.dist_pos_tracker.update_state(dist_pos)
        self.dist_neg_tracker.update_state(dist_neg)
        
        return {"loss": self.loss_tracker.result(), "dist_pos": self.dist_pos_tracker.result(), "dist_neg": self.dist_neg_tracker.result()}
                                        
    
#     def fit_siamese(self, data):
#         for batch in data :
#             self.train_step(batch)
#         return 0
        