import numpy as np
import pandas as pd
import cv2
import os
import sys
import math
import time
from sklearn import svm

labels = [(f) for f in os.listdir('./train')]
# sift提取的特征数量
nfeatures = 200
# 词袋中的center数量
ncenters = 100


def cal_vec(filepath):
    vecs = []
    label = []
    centers = np.load('./results/svm_centers_%d_%d.npy' % (nfeatures, ncenters))
    print("cal_vec start")
    for idx, f in enumerate(labels):
        print('cal_vec process:%d/%d' % (idx + 1, len(labels)))
        path = os.path.join(filepath, f)
        files = [(os.path.join(path, file)) for file in os.listdir(path)]
        for filename in files:
            img = cv2.imread(filename)
            img = segment_plant(img)
            # 对每张图片计算sift特征
            sift = cv2.SIFT_create(nfeatures)
            kp, des = sift.detectAndCompute(img, None)
            if des is not None:
                # 根据sift特征计算对应的vec
                vec = calcFeatVec(des, centers)
                vecs.append(vec.flatten())
                label.append(idx)
    print("cal_vec done")
    return vecs, label


def learnVocabulary(features):
    wordCnt = ncenters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    # 得到k-means聚类的中心点centers
    compactness, labels, centers = cv2.kmeans(features, wordCnt, None, criteria, 20, flags)
    return centers


def calcFeatVec(features, centers):
    featVec = np.zeros((1, ncenters))
    for i in range(0, features.shape[0]):
        fi = features[i]
        diffMat = np.tile(fi, (ncenters, 1)) - centers
        # 求特征到每个中心点的距离
        dist = ((diffMat ** 2).sum(axis=1)) ** 0.5
        # 取出最小的距离，即找到最近的中心点
        idx = np.argmin(dist)
        # 该中心点对应+1
        featVec[0][idx] += 1
    return featVec


# 利用训练集所有图片的sift特征构建词袋
def build_center(filepath):
    features = []
    print("build_center start")
    for idx, f in enumerate(labels):
        print('build_center process:%d/%d' % (idx + 1, len(labels)))
        path = os.path.join(filepath, f)
        files = [(os.path.join(path, file)) for file in os.listdir(path)]
        for filename in files:
            img = cv2.imread(filename)
            img = segment_plant(img)
            sift = cv2.SIFT_create(nfeatures)
            kp, des = sift.detectAndCompute(img, None)
            if des is not None:
                for d in des:
                    features.append(d.astype(np.float32))
    features = np.array(features)
    centers = learnVocabulary(features)
    filename = "./results/svm_centers_%d_%d.npy" % (nfeatures, ncenters)
    np.save(filename, centers)
    print("build_center done")


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
    build_center('./train')
    centers = np.load('./results/svm_centers_%d_%d.npy' % (nfeatures, ncenters))
    xtrain, ytrain = cal_vec('./train')
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)
    # 利用vec和label构建svm
    print('build_svm start')
    clf = svm.SVC(kernel='rbf')
    clf.fit(xtrain, ytrain)
    print('build_svm done')
    sift = cv2.SIFT_create()
    files_test = [(os.path.join('./test', f)) for f in os.listdir('./test')]
    pred_label = []
    for filename in files_test:
        img = cv2.imread(filename)
        img = segment_plant(img)
        kp, des = sift.detectAndCompute(img, None)
        vec = calcFeatVec(des, centers)
        temp = clf.predict(vec)
        pred = temp.astype(np.int64)
        counts = np.bincount(pred)
        pred_label.append(labels[np.argmax(counts)])

    img_test = [(f) for f in os.listdir('./test')]
    dataframe = pd.DataFrame({'file': img_test, 'species': pred_label})
    dataframe.to_csv('./results/SIFT+SVM_%d_%d' % (nfeatures, ncenters), index=False, sep=',')


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("\n--- %s seconds ---" % (time.time() - start_time))
