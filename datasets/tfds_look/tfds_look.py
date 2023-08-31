"""tdfs_mnist dataset."""

import tensorflow_datasets as tfds
import os
import skimage.io as io


class TFDS_LOOK(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for tdfs_mnist dataset."""
    
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    
    
    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""    #                        
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({            
                'image-anchor': tfds.features.Image(shape=(None, None, 3)),
                'image_positive': tfds.features.Image(shape=(None, None, 3)),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image-anchor', 'image-positive'),  
        )
    
    def _split_generators(self, dl_manager: tfds.download.DownloadManager):           
        self.path = '/mnt/hd-data/Datasets/Totally-Looks-Like-Data-20230824T145910Z-001/Totally-Looks-Like-Data'    #path to data    
        imcodes_train = os.path.join(self.path, 'list_codes_train.txt')
        imcodes_test = os.path.join(self.path, 'list_codes_test.txt')
        return {
            'train': self._generate_examples(os.path.join(self.path,imcodes_train)),                    
            'test': self._generate_examples(os.path.join(self.path,imcodes_test)),
            }
    
    def _generate_examples(self, fname):
        """Yields examples."""
        # TODO(tdfs_mnist): Yields (key, example) tuples from the dataset
        with open(fname) as fcodes :
            for imcode in fcodes : 
                imcode = imcode.strip()                                                                                  
                fimage_a = os.path.join(self.path, 'right', imcode + '.jpg')
                fimage_p = os.path.join(self.path, 'left', imcode + '.jpg')                
                image_a = io.imread(fimage_a)
                image_p = io.imread(fimage_p)                                
                yield imcode, {
                    'image-anchor': image_a,
                    'image_positive': image_p,
                }
                
          
