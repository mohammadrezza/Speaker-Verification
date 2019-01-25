import scipy.io.wavfile as wav
import progressbar
from proj_paths import *
from SVM.feature_extraction import extract
from utility import *


if __name__ == "__main__":

    bar = progressbar.ProgressBar(maxval=len(collect_files(SVM_DATA_SET_PATH)),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    print("Extracting features ...")
    features = []

    for i,(raw_file_name, joined_file_path) in enumerate(collect_files(SVM_DATA_SET_PATH)):
        _, sig = wav.read(joined_file_path)
        feats = extract(sig)
        features.append(feats)
        bar.update(i + 1)

    bar.finish()
    print("Done.")

