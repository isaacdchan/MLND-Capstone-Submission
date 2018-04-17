import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from utils_old import coord_generator, des_list, best_score_label, average_score_label
# import modules

img1 = cv2.imread('images\\img_test1.jpg', 0)
# import testing image
orb = cv2.ORB_create()
kpX, desX = orb.detectAndCompute(img1, None)
# import orb object (augemented form of SIFT) and store KPs and descriptors in the KPX and desX arrays


desX_list = []
for des in desX:
    desX_list.append(des)
# desX currently in immutable numpy array. appending values to list

kp_coord = coord_generator(kpX)
# propietary function that extracts and  stores the pixel coordiantes of each opencv KP object
clf = LocalOutlierFactor(n_neighbors=15)
y_pred = clf.fit_predict(kp_coord)
# filtering out SIFT features found on salient objects that aren't minions (projectiles)

for idx, val in sorted(np.ndenumerate(y_pred), reverse=True):
    if val == -1:
        del(kpX[idx[0]])
        del(desX_list[idx[0]])
# deleting outlier KPs

kp_coord2 = coord_generator(kpX)
# creating a new list of pixel coordinates based on the outlier free kpX list
kmeans = KMeans(n_clusters=11, random_state=0).fit(kp_coord2)
# creating a kmeans object with n_clusters hard coded
label_list = kmeans.labels_
center_list = kmeans.cluster_centers_
center_list = np.array([np.array(xi, dtype=int) for xi in center_list])
# creating a list of each KPs respective cluster and a list of each cluster's centroid coordinates

kp_label_data = zip(kpX, label_list)
kp_label_data = set(kp_label_data)


centroid = 1
# select the specific minion on screen that you which to match
cluster_kp_list = []
cluster_des_list = []


for idx, value in enumerate(kp_label_data):
    if value[1] == centroid:
        cluster_kp_list.append(value[0])
        cluster_des_list.append(desX_list[idx])
# extract KPs and descriptors from KPX if they belong to the desired cluster


cluster_des_list = np.array([np.array(xi) for xi in cluster_des_list])
# converting the list of descriptors back to an array for opencv
score_list = []

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

for i in des_list:
    matches = bf.match(cluster_des_list, i)
    matches = sorted(matches, key=lambda x: x.distance)
# des_list is a list of arrayscontatining the descriptor values of each of the 25 template images
# this loop matches the extracted descriptor values of the desired against each of the 25 images

    score = 0
    for i in range(10):
        score += (matches[i].distance)
    score_list.append(score / 10)
# finds the average score of the top 10 closest matches and appends it to score list for each of the 25 images

score_dict = dict()
for index, score in np.ndenumerate(score_list):
    score_dict[index] = score
# creating a dictionary where the key is the template images' number and the value is its score

closest_match = min(score_dict, key=score_dict.get)[0]
# proprietary function that returns the average score of each minion type comapred to the test image

print('cluster:', centroid)
print('closest single match:', closest_match, best_score_label(closest_match))
print('best average score:', average_score_label(score_list))
