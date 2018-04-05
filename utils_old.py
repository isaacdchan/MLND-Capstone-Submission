from kp_generator import des1, des2, des3, des4, des5, des6, des7, des8, des9, des10, \
    des11, des12, des13, des14, des15, des16, des17, des18, des19, des20, des21, des22, des23, \
    des24, des25
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

des_list = [des1, des2, des3, des4, des5, des6, des7, des8, des9, des10, \
    des11, des12, des13, des14, des15, des16, des17, des18, des19, des20, des21, des22, des23, \
    des24, des25]

def best_score_label(closest_match):
    if closest_match < 5:
        return 'blue cannon'
    elif closest_match < 8:
        return 'blue melee'     
    elif closest_match < 12:
        return 'blue ranged'    
    elif closest_match < 16:
        return 'red cannon'          
    elif closest_match < 21:
        return 'red melee'   
    else:
        return 'red ranged'

def average_score_label(score_list):
    average_score_list = []
    average_score_list.append(sum(score_list[0:5])/5)
    average_score_list.append(sum(score_list[5:8])/3)
    average_score_list.append(sum(score_list[8:12])/4)
    average_score_list.append(sum(score_list[12:16])/4)
    average_score_list.append(sum(score_list[16:21])/5)
    average_score_list.append(sum(score_list[21:25])/4)

    best_index, best_value = min(enumerate(average_score_list), key=operator.itemgetter(1))
    # if best_index == 0: return 'blue cannon'
    # if best_index == 1: return 'blue melee'
    # if best_index == 2: return 'blue ranged'
    # if best_index == 3: return 'red cannon'
    # if best_index == 4: return 'red melee'
    # if best_index == 5: return 'red ranged'
    return average_score_list
    return best_index


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

    im = Image.open(img_path)
    pix = im.load()
    print(im.size)

    roi = cluster_roi()
    x = centroid[0]
    y = centroid[1]

    return im[(x-50):(x+50), (y-50):(y+50)]
