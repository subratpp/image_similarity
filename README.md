# Image Similarity Determination

The codes are written in Python running on Ubuntu 20. So use Linux to execute the codes.

The idea is to get the features of given image using pretrained model. Then compute cosine similarity with new images. Another idea is to build a apple vs non-apple classifier but that will require more images to train. Transfer learning as well could be used to train last few layers to classify apple from non-apple images. I went with the first appraoch.

## Dependencies Packages Python

-	Pytorch
-	Sklearn
-	PIL


## System Setup

Install linux package by running following commands. This package helps to get notification of any files added to test directory.
```
sudo apt-get install inotify-tools
```

# Execution Information

1. Install packages
2. Run the script file `script.sh` from linux terminal. Add file to `test` directory to check working of codes.

- The Jupyter Notebook file `feature_extraction_resnet18.ipynb` extracts the features of given images of apple in `sample_image` folder. The feature vectors are saved in `embedded.csv` file. So, no need to re-run this file. The .csv file is loaded whenever new image is added to `test` folder and cosine similarity is computed on the image with respect to given apple images. If the score is above 0.7, the images are consider similar.
- The python file `check_similarity.py` is called whenever new file is added to directory `test`.
- Tkinter package is used to display a dialog box. But the code is commented out.