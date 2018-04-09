import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
from utils import coord_generator, cluster_roi, screen_cluster_roi
from PIL import ImageGrab
import time
import operator


def screen_record():
    last_time = time.time()
# tracks how long each iteration of algorithm takes. essentially frames per second (fps)
    while(True):
        screen = np.array(ImageGrab.grab(bbox=(0, 100, 1100, 800)))
# screencapture function where 3rd and 4th number determine resolution
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(screen, None)


        des2 = []
        for des in des1:
            des2.append(des)


        kp_coord = coord_generator(kp1)
        clf = LocalOutlierFactor(n_neighbors=15)
        y_pred = clf.fit_predict(kp_coord)


        for idx, val in sorted(np.ndenumerate(y_pred), reverse=True):
            if val == -1:
                del(kp1[idx[0]])
                del(des2[idx[0]])
                del(kp_coord[idx[0]])
# for explanation on lines 18-36 see cluster_bfmatch.py

        silhouette_dict = {}
# creating dictionary of silhouette values to determine which n_clusters value was best
        for i in range(2,15):
            kmeans = KMeans(n_clusters=i, n_init=1, random_state=0).fit(kp_coord)
# run kmeans 15 times in range(2, 15) to determine actual number of minions on screen

            labels = kmeans.labels_
            centers = kmeans.cluster_centers_

            silhouette_dict[i] = metrics.silhouette_score(kp_coord, labels)

        best_n = max(silhouette_dict.items(), key=operator.itemgetter(1))[0]
# retrun best scoring value of n_clusters
        kmeans = KMeans(n_clusters= best_n, random_state=0).fit(kp_coord)
# run kmeans one final time with the best n_cluster value
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_


        font = cv2.FONT_HERSHEY_SIMPLEX


        for idx, val in enumerate(centers):
            rasterized_cluster = screen_cluster_roi(screen, centers[idx])
# using the list of centroid centers, use a proprietary function to rasterize a bounding box around each centroid

            rasterized_cluster2 = cv2.cvtColor(rasterized_cluster, cv2.COLOR_BGR2RGB)
            try:
                hsv = cv2.cvtColor(rasterized_cluster2 ,cv2.COLOR_RGB2HSV)

                lower_range = np.array([0,165,0])
                upper_range = np.array([255,255,255])

                mask = cv2.inRange(hsv, lower_range, upper_range)
                masked = cv2.bitwise_and(rasterized_cluster, rasterized_cluster, mask=mask)


                masked2 = masked.reshape(10000, 3)
                masked2 = np.sum(masked2, axis=0)

                masked3 = masked[np.nonzero(masked)]
# filter out low saturation values, leaving only the red and blue of a minion left in the image

                averages = np.divide(masked2, masked3.size)

                if averages[0] > averages[2]:
                    cv2.putText(screen, 'Red', (int(val[0]), int(val[1])), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(screen, 'Blue', (int(val[0]), int(val[1])), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
# average the RGB values of the remaining red and blue pixels, if R is higher than B, label minion as red, vice versa
            except:
                pass
        
        print('Loop took {} seconds'.format(time.time() - last_time))
        last_time = time.time()

        cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

screen_record()