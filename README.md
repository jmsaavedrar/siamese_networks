# Siamese Neworks
This repo implements a resnet-based siamese network for representation learning.  You can run th project using the following steps ;

## Prepare the dataset for training and testing
   For this example we use the [Tootally-Looks-Like](https://sites.google.com/view/totally-looks-like-dataset)  dataset.  To use it, we recommend using tfds [tfds_look.py](../datasets/tfds_look/tfds_look.py). We split this dataset into 
   5000 image pairs for training and 116 for testing. You can dowloand the tfds format of the dataset from [here](https://www.dropbox.com/scl/fi/kbcntw4rb4vstf19caaer/tfds_look.tar?rlkey=xlx1sycgsruee89x1lyei5d0m&dl=0). Please download it and apply tar -xf into ~/tensorflow_datasets.
   
   
   
   
## Train the siamese_network using the tfds_look
python train.py -config config/look.ini -model RESNET -gpu 0
You can download the trained model from [here]().

## Test the trained model
We will apply similarity search to test the trained encoder. To this end, we should use the sim_search project. 

1) Download the [testing images] (https://www.dropbox.com/scl/fi/qb50cb9lw1umxd5loeb3v/images_for_test.zip?rlkey=o2jbn08ozj5iwgnfc9o9phl32&dl=0)
2) Generate two files one with the name of the images in folder right and the other with those in the left. 

3) Run

python ssearch.py 
		-config <SIAMESE_NETWORK FOLDER>config/look.ini
		-catalog <FILE WITH CATALOG IMAGES> 
		-query <FILE WITH QUERY IMAGES>
		-model_path  <MODEL PATH>


some results will be saved in the folder named results.# siamese_networks_
