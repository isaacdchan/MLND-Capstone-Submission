import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from utils_old import coord_generator, des_list, best_score_label, average_score_label

img1 = cv2.imread('images\\img_test1.jpg', 0)

orb = cv2.ORB_create()
kpX, desX = orb.detectAndCompute(img1, None)


desX_list = []
for des in desX:
    desX_list.append(des)


kp_coord = coord_generator(kpX)
clf = LocalOutlierFactor(n_neighbors=15)
y_pred = clf.fit_predict(kp_coord)


for idx, val in sorted(np.ndenumerate(y_pred), reverse=True):
    if val == -1:
        del(kpX[idx[0]])
        del(desX_list[idx[0]])


kp_coord2 = coord_generator(kpX)
kmeans = KMeans(n_clusters=11, random_state=0).fit(kp_coord2)
label_list = kmeans.labels_
center_list = kmeans.cluster_centers_
center_list = np.array([np.array(xi, dtype=int) for xi in center_list])


kp_label_data = zip(kpX, label_list)
kp_label_data = set(kp_label_data)


centroid = 1
cluster_kp_list = []
cluster_des_list = []


for idx, value in enumerate(kp_label_data):
    if value[1] == centroid:
        cluster_kp_list.append(value[0])
        cluster_des_list.append(desX_list[idx])


cluster_des_list = np.array([np.array(xi) for xi in cluster_des_list])
score_list = []

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

for i in des_list:
    matches = bf.match(cluster_des_list, i)
    matches = sorted(matches, key=lambda x: x.distance)

    score = 0
    for i in range(10):
        score += (matches[i].distance)
    score_list.append(score / 10)


score_dict = dict()
for index, score in np.ndenumerate(score_list):
    score_dict[index] = score
    

closest_match = min(score_dict, key=score_dict.get)[0]

print('cluster:', centroid)
print('closest single match:', closest_match, best_score_label(closest_match))
print('best average score:', average_score_label(score_list))