import numpy as np
import pandas as pd
import cv2
import os
import sys
import math
import time
from sklearn import svm
from skimage.feature import hog

labels = [(f) for f in os.listdir('./train')]

def HOG(filepath):
    features = []
    label = []
    print("HOG_training start")
    for idx, f in enumerate(labels):
        print('HOG_training process:%d/%d' % (idx + 1, len(labels)))
        path = os.path.join(filepath, f)
        files = [(os.path.join(path, file)) for file in os.listdir(path)]
        for filename in files:
            img = cv2.imread(filename)
            img = segment_plant(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (128, 128))
            # 对每张图片计算hog特征
            des = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(8, 8), block_norm='L2')
            features.append(des)
            label.append(idx)
    print("HOG_training done")
    return features, label


def create_mask_for_plant(image):
    # bgr转化为hsv
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 设置阈值
    lower_hsv = np.array([25, 40, 40])
    upper_hsv = np.array([80, 255, 255])
    # 二值化
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    return mask


def segment_plant(image):
    mask = create_mask_for_plant(image)
    # 求交集 、 掩膜提取图像
    output = cv2.bitwise_and(image, image, mask=mask)
    return output


def main():
    xtrain, ytrain = HOG('./train')
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)
    # 利用vec和label构建svm
    print('build_svm start')
    clf = svm.SVC(kernel='rbf')
    clf.fit(xtrain, ytrain)
    print('build_svm done')
    files_test = [(os.path.join('./test', f)) for f in os.listdir('./test')]
    pred_label = []
    for filename in files_test:
        img = cv2.imread(filename)
        img = segment_plant(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 128))
        des = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(8, 8), block_norm='L2')
        des = des.reshape(1,-1)
        pred = clf.predict(des)
        pred_label.append(labels[int(pred)])

    img_test = [(f) for f in os.listdir('./test')]
    dataframe = pd.DataFrame({'file': img_test, 'species': pred_label})
    dataframe.to_csv('./results/HOG+SVM_16_8_rbf.csv', index=False, sep=',')


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("\n--- %s seconds ---" % (time.time() - start_time))
