# Siamese Neworks
## Prepare the dataset for training and testing
   Here, we recommend to use tfds (tfds_look.py)
   train : 5000
   test : 116
   download [tfds_look.tar](https://www.dropbox.com/scl/fi/kbcntw4rb4vstf19caaer/tfds_look.tar?rlkey=xlx1sycgsruee89x1lyei5d0m&dl=0) and untar it into ~/tensorflow_datasets
   
   
   
   
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
