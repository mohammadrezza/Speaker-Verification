from python_speech_features import sigproc, mfcc, delta
from settings import *


def extract(sig):
    # framing
    sig_frames = sigproc.framesig(sig=sig, frame_len=FRAME_LENGTH, frame_step=FRAME_STEP)
    feat = []

    def calc_all_feat(feat_coeffs):
        feat.extend(feat_coeffs.max(axis=0))
        feat.extend(feat_coeffs.min(axis=0))
        feat.extend(feat_coeffs.mean(axis=0))
        feat.extend(feat_coeffs.var(axis=0))

    # region calculate mfcc features
    mfcc_feat = mfcc(signal=sig_frames, samplerate=SAMPLE_RATE, winlen=WINDOW_LENGTH, winstep=WINDOW_STEP,
                     numcep=13, preemph=PRE_EMPH, winfunc=WINDOW_FUNCTION)
    mfcc_feat_delta = delta(mfcc_feat, 20)
    mfcc_feat_delta_delta = delta(mfcc_feat_delta, 20)

    calc_all_feat(mfcc_feat)
    calc_all_feat(mfcc_feat_delta)
    calc_all_feat(mfcc_feat_delta_delta)

    # endregion

    # region calculate zero cross rating
    def zcr(frames):
        def sign(x):
            return 1 if x >= 0 else -1

        zc_rates = []
        for frame in frames:
            zc_rate = 0
            for i in range(1, len(frame)):
                zc_rate += abs(sign(frame[i]) - sign(frame[i - 1])) / 2
            zc_rates.append(zc_rate / len(frame))
        return zc_rates

    zcrs = zcr(sig_frames)
    calc_all_feat(np.array([zcrs]).reshape(len(zcrs), 1))

    # endregion

    # region calculate energy
    def energy(frames, eta):
        energys = []
        for frame in frames:
            energy = 1 / len(frame) * np.sum(np.power(frame, 2))
            energys.append(energy)
        return energys

    energys = energy(sig_frames, 0)
    calc_all_feat(np.array([energys]).reshape(len(energys), 1))
    # endregion

    return feat
