# Image Similarity Determination

The codes are written in Python running on Ubuntu 20. So use Linux to execute the codes.

The idea is to get the features of a given image using a pre-trained model. Then compute cosine similarity with new images. Another idea is to build an apple vs non-apple classifier but that will require more images to train. Transfer learning as well could be used to train the last few layers to classify apples from non-apple images. I went with the first approach.

## Dependencies Packages Python

-	Pytorch
-	Sklearn
-	PIL


## System Setup

Install Linux package by running the following commands. This package helps to get notification of any files added to the 'test' directory.
```
sudo apt-get install inotify-tools
```

# Execution Information

1. Install packages
2. Run the script file `script.sh` from the Linux terminal. Add file to `test` directory to check the working of code.

- The Jupyter Notebook file `feature_extraction_resnet18.ipynb` extracts the features of given images of apple in `sample_image` folder. The feature vectors are saved in `embedded.csv` file. So, no need to re-run this file. The .csv file is loaded whenever a new image is added to `test` folder and cosine similarity is computed on the image concerning given apple images. If the score is above 0.7, the images are considered similar.
- The python file `check_similarity.py` is called whenever a new file is added to the directory `test`.
- Tkinter package is used to display a dialogue box. But the code is commented out.
