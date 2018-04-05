import operator
import numpy as np
import cv2
from PIL import Image

def coord_generator(kp_list):

    kp_coord = []
    for kp in kp_list:
        kp_temp = []
        x1, y1 = kp.pt

        kp_temp.append(x1)
        kp_temp.append(y1)
        kp_coord.append(kp_temp)

    return kp_coord


def centroid_histogram(kmeans):
    num_labels = np.arange(0, len(np.unique(kmeans.labels_)) + 1)
    (hist, _) = np.histogram(kmeans.labels_, bins = num_labels)
 
    hist = hist.astype("float")
    hist /= hist.sum()
 
    return hist


def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0
 
    for (percent, color) in zip(hist, centroids):

        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX
    
    return bar


def cluster_roi(img_path, centroid):

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    y = centroid[0]
    x = centroid[1]

    rasterized_cluster = img[int(x-50):int(x+50), int(y-50):int(y+50)]
    return rasterized_cluster

def screen_cluster_roi(screen, centroid):

    y = centroid[0]
    x = centroid[1]

    rasterized_cluster = screen[int(x-50):int(x+50), int(y-50):int(y+50)]
    return rasterized_cluster

def hp_roi(img_path, centroid):

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    x = centroid[0]
    y = centroid[1]

    rasterized_cluster = img[int(y-115):int(y-15), int(x-40):int(x+140)]
    return rasterized_cluster

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
 
    # return the histogram
    return hist

def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0
 
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX
    
    # return the bar chart
    return bar