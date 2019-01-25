from fastdtw import fastdtw, dtw
from scipy.spatial.distance import euclidean
import scipy.io.wavfile as wav
from DTW.extract_features import extract
from utility import *
from proj_paths import *

models = dict()


def load_models():
    for raw_model_name, model_path in collect_files(DTW_MODELS_PATH):
        models[raw_model_name] = load(model_path)


def real_time():
    while True:
        try:
            for raw_file, joined_file in collect_files(REAL_TIME_PATH):
                _, sig = wav.read(joined_file)
                test_feats = extract(sig)
                results = dict()
                for model_name, model_feats in models.items():
                    dist, path = dtw(test_feats, model_feats, euclidean)
                    print(dist)
                os.remove(joined_file)
        except Exception as e:
            # print(e.__str__())
            pass


if __name__ == "__main__":
    load_models()
    real_time()
