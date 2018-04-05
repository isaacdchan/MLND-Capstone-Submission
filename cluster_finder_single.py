import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from utils_old import coord_generator

img1 = cv2.imread('images\\img_test2.jpg', 1)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)


kp_coord = coord_generator(kp1)
clf = LocalOutlierFactor(n_neighbors=15)
y_pred = clf.fit_predict(kp_coord)


for idx, val in sorted(np.ndenumerate(y_pred), reverse=True):
    if val == -1:
        del(kp1[idx[0]])


kp_coord2 = coord_generator(kp1)
kmeans = KMeans(n_clusters=11, random_state=3).fit(kp_coord2)
label_list = kmeans.labels_
center_list = kmeans.cluster_centers_
center_list = np.array([np.array(xi, dtype=int) for xi in center_list])


kp_label_data = zip(kp1, label_list)
kp_label_data = set(kp_label_data)


centroid = 3


cluster_kp_list = []
for kp, cluster in kp_label_data:
    if cluster == centroid:
        cluster_kp_list.append(kp)



img1 = cv2.circle(img1, (center_list[centroid][0], center_list[centroid][1]), 70, (0, 255, 0), 3)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

plt.imshow(img1)
plt.show()
