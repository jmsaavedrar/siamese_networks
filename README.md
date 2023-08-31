# Siamese Neworks
This repo implements a resnet-based siamese network for representation learning.  You can run th project using the following steps ;

## Prepare the dataset for training and testing
   For this example we use the [Tootally-Looks-Like](https://sites.google.com/view/totally-looks-like-dataset)  dataset.  To use it, we recommend converting the dataset into tfds format (see [tfds_look.py](../datasets/tfds_look/tfds_look.py)). We split this dataset into 
   5000 image pairs for training and 116 for testing. You can dowloand the tfds format of the dataset from [here](https://www.dropbox.com/scl/fi/kbcntw4rb4vstf19caaer/tfds_look.tar?rlkey=xlx1sycgsruee89x1lyei5d0m&dl=0). Please download it and apply tar -xf into ~/tensorflow_datasets.
   
      
   
## Train the siamese_network using the tfds_look

python train.py -config config/look.ini -model RESNET -gpu 0

You can download the trained model from [here](https://www.dropbox.com/scl/fi/m8j8hcsn4g9pzhbpeh04n/model_look.tar?rlkey=z7mg78nckhwffsv1zk3da1q65&dl=0). Apply tar -xf into the project folder.

## Test the trained model
We will apply similarity search to test the trained encoder. To this end, we can use the [sim_search project](https://github.com/jmsaavedrar/siamese_networks). Then apply the folowing steps:

 1.  Download the [testing images](https://www.dropbox.com/scl/fi/qb50cb9lw1umxd5loeb3v/images_for_test.zip?rlkey=o2jbn08ozj5iwgnfc9o9phl32&dl=0). The dataset contains two parallel folders (right and left).
 2. Generate two files,  one with the files in the folder **right** and the other with those in the **left**. 
 3. Run
	python ssearch.py 
		-config <SIAMESE_NETWORK FOLDER>config/look.ini
		-catalog <FILE WITH CATALOG IMAGES> 
		-query <FILE WITH QUERY IMAGES>
		-model_path  <MODEL PATH>


After running you will get some results in the folder **results**. Some examples appear below (first image is the query and the rest are ordered according to the similarity with the first):
![output_00204 jpg](https://github.com/jmsaavedrar/siamese_networks/assets/8441460/d5d9c114-7cb4-4908-9d93-e393e2043847)
![output_00518 jpg](https://github.com/jmsaavedrar/siamese_networks/assets/8441460/6eeb9a38-a34f-420d-b6c9-ee74b34251b8)
![output_00659 jpg](https://github.com/jmsaavedrar/siamese_networks/assets/8441460/1295e05a-e6f2-4a1e-9ea5-70daf5ab298c)
![output_00944 jpg](https://github.com/jmsaavedrar/siamese_networks/assets/8441460/a7d9d657-98af-4f3e-9645-4c211b6d08fd)
![output_01097 jpg](https://github.com/jmsaavedrar/siamese_networks/assets/8441460/34a806ea-50a2-47bf-a4dd-28cbcc56e38a)
![output_02537 jpg](https://github.com/jmsaavedrar/siamese_networks/assets/8441460/311939e4-e879-4f22-9b64-70a1f41f4a65)
