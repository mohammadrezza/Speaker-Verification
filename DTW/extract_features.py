from python_speech_features import sigproc, mfcc, delta
from settings import *


def extract(sig):
    # framing
    sig_frames = sigproc.framesig(sig=sig, frame_len=FRAME_LENGTH, frame_step=FRAME_STEP)
    frames_feats = None

    def concat_feats(feat_coeffs):
        return np.concatenate((frames_feats, feat_coeffs), axis=1)

    # region calculate mfcc features
    mfcc_feat = mfcc(signal=sig, samplerate=SAMPLE_RATE, winlen=WINDOW_LENGTH, winstep=WINDOW_STEP,
                     numcep=13, preemph=PRE_EMPH, winfunc=WINDOW_FUNCTION)
    mfcc_feat_delta = delta(mfcc_feat, 20)
    mfcc_feat_delta_delta = delta(mfcc_feat_delta, 20)

    frames_feats = mfcc_feat
    frames_feats = concat_feats(mfcc_feat_delta)
    frames_feats = concat_feats(mfcc_feat_delta_delta)

    # endregion

    # region calculate zero cross rating
    def zcr(frames):
        def sign(x):
            return 1 if x >= 0 else -1

        zcrs = []
        for frame in frames:
            zc_rate = 0
            for i in range(1, len(frame)):
                zc_rate += abs(sign(frame[i]) - sign(frame[i - 1])) / 2
            zcrs.append(zc_rate / len(frame))
        return zcrs

    zcrs = zcr(sig_frames)
    frames_feats = concat_feats(np.array([zcrs]).reshape(len(zcrs), 1))

    # endregion

    # region calculate energy
    def autocorrelate(frames, eta):
        energys = []
        for frame in frames:
            total_sum = 0
            for i in range(eta, len(frame)):
                total_sum += frame[i] * frame[i - eta]
            energy = 1 / len(frame) * total_sum
            energys.append(energy)
        return energys

    energys = autocorrelate(sig_frames, 0)
    frames_feats = concat_feats(np.array([energys]).reshape(len(energys), 1))
    # endregion

    # frames_feats = frames_feats/frames_feats.max(axis=1).reshape(frames_feats.shape[0],1)
    return frames_feats/100000
