import os
from collections import defaultdict

import numpy as np
import cv2

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

if __name__ == "__main__":
    bcf = BCF()
    bcf.load_classes()
    bcf.load_training()
    for cls in bcf.training_images:
        bcf.training_images[cls] = bcf.normalize_shapes(bcf.training_images[cls])
        bcf.extract_cf(bcf.training_images[cls])

