# MLND Capstone

Final capstone project for Udacity's Machine Learning Nanodegree.

### Labeling Standardized Videogame Objects Using Computer Vision and Clustering

In this project I attempt to take advantage of the predictability and uniformity of objects and video games to train a classifier that doesn't require exorbitant amounts of data.

## Getting Started

The pdf's included in this repor are my initial project proposal and a report describing my methodology and results. 
screen_label.py is the final classifier script that uses functions stored in the utils.py file
cluster_bfmatch.py, cluster_finder_single.py, kp_generator.py, utils.py are all scripts for the computer vision based benchmark model I created
The images folder contains the testing images I used to test accuracy and tweak screen_label.py. It also contains a dictionary of template images used by the benchmark model
Let me know if you run into any bugs and I can try to walk you through it

### Prerequisites
* Python 3.5 or higher
* OpenCV (known as cv2 once downloaded) - opencv-python if installing from command line
* matplotlib
* numpy
* scikit-learn
* PIL (python imaging library)
* operator
