import os
from collections import defaultdict

from scipy.spatial.distance import cdist
import numpy as np
import cv2

from evolution import evolution

class BCF():
    def __init__(self):
        self.DATA_DIR = "data/cuauv"
        self.PERC_TRAINING_PER_CLASS = 0.5
        self.classes = defaultdict(list)
        self.training_images = defaultdict(list)
        self.normalized_training_images = defaultdict(list)

    def load_classes(self):
        for dir_name, subdir_list, file_list in os.walk(self.DATA_DIR):
            if subdir_list:
                continue
            for f in file_list:
                self.classes[dir_name.split('/')[-1]].append(os.path.join(dir_name, f))

    def load_training(self):
        for cls in self.classes:
            images = self.classes[cls]
            self.training_images[cls] = [cv2.imread(image, cv2.IMREAD_GRAYSCALE) for image in
                                         images[:int(len(images) * self.PERC_TRAINING_PER_CLASS)]]

    def normalize_shapes(self, images):
        normalized = []
        for image in images:
            # Remove void space
            y, x = np.where(image > 50)
            max_y = y.max()
            min_y = y.min()
            max_x = x.max()
            min_x = x.min()
            trimmed = image[min_y:max_y, min_x:max_x] > 50
            trimmed = trimmed.astype('uint8')
            trimmed[trimmed > 0] = 255
            normalized.append(trimmed)
        return normalized

    def extract_cf(self, images):
        for image in images:
            contour, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mat = np.zeros(image.shape, np.int8)
            cv2.drawContours(mat, contour, -1, (255, 255, 255))

    def learn_codebook(self):
        pass

    def encode_cf(self):
        pass

    def spp(self):
        pass

    def svm_classify(self):
        pass

    def show(self, image):
        cv2.imshow('image', image)
        _ = cv2.waitKey()

    def extr_raw_points(self, c, max_value, N, nn):
        # -------------------------------------------------------
        # [SegmentX, SegmentY,NO]=GenSegmentsNew(a,b,maxvalue,nn)
        # This function is used to generate all the segments
        # vectors of the input contour
        # a and b are the input contour sequence
        #  maxvalue is the stop condition of DCE, usually 1~1.5
        #  nn is the sample points' number on each segment, in super's method,n=25
        # SegmentX,SegmentY denotes all the coordinates of all the segments of input contour
        # NO denotes the number of segments of input contour
        # -------------------------------------------------------
        kp, _, _ = evolution(c, N, max_value, 0, 0, 0) # critical points
        n2 = cdist(kp, c)

        i_kp = np.argmin(n2.transpose(), axis=0) # column-wise min
        n_kp = len(i_kp)
        n_cf = (n_kp - 1) * n_kp + 1
        pnts = [None] * n_cf

        s = 0
        for i in range(n_kp):
            for j in range(n_kp):
                if i == j:
                    continue
                if i < j:
                    cf = c[i_kp[i]:i_kp[j]+1, :]
                if i > j:
                    cf = np.append(c[i_kp[i]:, :], c[:i_kp[j]+1, :], axis=0)
                pnts[s] = self.sample_contour(cf, nn)
                s += 1
        pnts[s] = self.sample_contour(c, nn)
        return pnts

    def sample_contour(self, cf, nn):
        # Sample points from contour fragment
        _len = cf.shape[0]
        ii = np.round(np.arange(0, _len - 0.9999, float(_len - 1) / (nn - 1))).astype('int32')
        cf = cf[ii, :]
        return cf

if __name__ == "__main__":
    bcf = BCF()
    #bcf.load_classes()
    #bcf.load_training()
    #for cls in bcf.training_images:
    #    bcf.training_images[cls] = bcf.normalize_shapes(bcf.training_images[cls])
    #    bcf.extract_cf(bcf.training_images[cls])
    C = np.array([
    [6.0000,    5.8000],
    [8.4189,    5.8000],
   [10.8378,    5.8000],
   [12.8425,    6.8000],
   [15.2614,    6.8000],
   [17.6803,    6.8000],
   [19.7773,    7.5773],
   [22.1039,    7.8000],
   [24.1086,    6.8000],
   [26.5275,    6.8000],
   [28.9464,    6.8000],
   [30.9511,    7.8000],
   [33.2616,    8.0616],
   [35.3747,    8.8000],
   [37.3794,    9.8000],
   [39.7983,    9.8000],
   [42.1536,    9.6464],
   [43.8640,    7.9360],
   [46.1602,    7.9602],
   [48.2313,    8.8000],
   [50.4597,    9.2597],
   [52.6548,    9.8000],
   [54.6595,   10.8000],
   [57.0555,   10.8555],
   [59.0831,   11.8000],
   [61.0878,   12.8000],
   [63.3583,   13.1583],
   [65.3616,   13.4384],
   [67.1019,   11.8000],
   [69.5208,   11.8000],
   [71.9397,   11.8000],
   [73.9444,   12.8000],
   [75.9640,   13.7640],
   [78.3680,   13.8000],
   [80.3727,   14.8000],
   [82.5597,   15.3597],
   [84.7963,   15.8000],
   [86.8593,   16.6593],
   [89.1555,   16.9555],
   [91.1588,   17.9588],
   [92.2000,   19.9464],
   [93.2000,   21.9511],
   [93.2000,   24.3700],
   [94.2000,   26.3747],
   [94.7611,   28.5611],
   [95.2000,   30.7983],
   [94.0657,   32.2000],
   [91.6468,   32.2000],
   [89.2279,   32.2000],
   [86.8090,   32.2000],
   [84.8616,   31.0616],
   [82.7996,   30.2000],
   [80.5621,   29.7621],
   [78.5587,   28.7587],
   [76.3713,   28.2000],
   [74.3666,   27.2000],
   [73.2000,   28.9209],
   [70.9430,   29.2000],
   [68.6635,   28.8635],
   [66.6602,   27.8602],
   [64.5147,   27.2000],
   [62.5100,   26.2000],
   [60.5053,   25.2000],
   [58.3540,   24.5540],
   [56.4960,   23.2000],
   [54.3474,   22.5474],
   [52.0724,   22.2000],
   [50.0479,   21.2479],
   [48.0445,   20.2445],
   [46.2000,   21.2245],
   [45.6394,   23.2000],
   [43.2205,   23.2000],
   [41.2158,   22.2000],
   [39.2111,   21.2000],
   [37.1460,   20.3460],
   [35.1426,   19.3426],
   [33.1393,   18.3393],
   [30.8431,   18.0431],
   [28.7734,   17.2000],
   [26.7687,   18.2000],
   [24.5403,   17.7403],
   [22.5370,   16.7370],
   [20.7547,   15.2000],
   [18.7500,   14.2000],
   [16.7453,   13.2000],
   [14.7406,   12.2000],
   [12.7359,   11.2000],
   [10.8099,   10.0099],
    [8.7265,    9.2000],
    [6.8033,    8.0033]
    ])
    max_curvature = 1.5
    n_contsamp = 50
    n_pntsamp = 10
    print bcf.extr_raw_points(C, max_curvature, n_contsamp, n_pntsamp)

