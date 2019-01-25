from sklearn.svm import SVC
from sklearn.decomposition import PCA
import scipy.io.wavfile as wav
import numpy as np
from utility import *
from proj_paths import *
from SVM.feature_extraction import extract


def pca_fit(x):
    pca_obj.fit(np.array(x))


def pca_transform(x):
    pca_x = pca_obj.transform(np.array(x))
    return pca_x


def classify(x, y):
    clf = SVC(C=1, gamma='auto', probability=True)
    clf.fit(np.array(x), np.array(y))
    return clf


def keep_predicting():
    # remove previous files
    for file in os.listdir(REAL_TIME_PATH):
        os.remove(os.path.join(REAL_TIME_PATH, file))
    while True:
        try:
            for file in os.listdir(REAL_TIME_PATH):
                rate, sig = wav.read(os.path.join(REAL_TIME_PATH, file))
                feat = extract(sig)
                pca_feats = pca_transform([feat])
                result = words_clf.predict(pca_feats)
                result2 = gender_clf.predict(pca_feats)

        except Exception as e:
            print(e.__str__())


if __name__ == "__main__":
    # reading the data from saved models in train
    features = load(os.path.join(MODELS_PATH, WORDS_FEATURES))
    words_labels = load(os.path.join(MODELS_PATH, WORDS_LABLES))
    gender_labels = load(os.path.join(MODELS_PATH, GENDER_LABLES))

    pca_obj = PCA(n_components=50, whiten=True)
    pca_fit(features)
    pca_feats = pca_transform(features)
    words_clf = classify(pca_feats, words_labels)
    gender_clf = classify(pca_feats, gender_labels)

    # print("words score :", words_clf.score(pca_feats, words_labels) * 100)
    # print("gender score :", gender_clf.score(pca_feats, gender_labels) * 100)

    # either keep predicting or test a folder
    # keep_predicting()
    test_a_folder(".\\Test")
