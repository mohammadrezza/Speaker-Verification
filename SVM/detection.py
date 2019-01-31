from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import scipy.io.wavfile as wav
import numpy as np
from utility import *
from proj_paths import *
from SVM.feature_extraction import extract
import os
import time


def pca_fit(x):
    pca_obj.fit(np.array(x))


def pca_transform(x):
    pca_x = pca_obj.transform(np.array(x))
    return pca_x


def classify(x):
    clf = OneClassSVM(kernel='rbf', degree=3, gamma='auto',
                      coef0=0.0, tol=1e-10, nu=0.3, shrinking=True, cache_size=400,
                      verbose=True, max_iter=-1, random_state=None)
    clf.fit(np.array(x))
    return clf


def keep_predicting():
    # remove previous files
    while True:
        try:
            for raw_file_name, joined_file_path in collect_files(REAL_TIME_PATH):
                rate, sig = wav.read(joined_file_path)
                feat = extract(sig)
                pca_feats = pca_transform([feat])

                score = clf.score_samples(pca_feats)
                print(raw_file_name, score)

                os.remove(joined_file_path)
        except Exception as e:
            # print(e.__str__())
            pass


def cal_threshold():
    scores = 0
    # remove previous files
    # while True:
    try:
        for raw_file_name, joined_file_path in collect_files(SVM_DATA_SET_PATH):
            rate, sig = wav.read(joined_file_path)
            feat = extract(sig)
            pca_feats = pca_transform([feat])

            score = clf.score_samples(pca_feats)
            scores += score[0]
            print(raw_file_name, score)

            # os.remove(joined_file_path)
    except Exception as e:
        # print(e.__str__())
        pass
    print(scores / 125)


if __name__ == "__main__":
    # reading the data from saved models in train
    features = load(SVM_FEATURES_NAME)
    pca_obj = PCA(n_components=30, whiten=True)
    pca_fit(features)
    pca_feats = pca_transform(features)
    labels = [1 for _ in range(len(features))]
    clf = classify(pca_feats)
    keep_predicting()
    time.sleep(1)
