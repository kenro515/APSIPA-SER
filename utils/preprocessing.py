import os
import glob
import numpy as np
import pyworld as pw
# import pysptk as sptk
from tqdm import tqdm
from scipy.io import wavfile

def pw_analyze(path_wav_analyze):
    fs, wav_data = wavfile.read(path_wav_analyze)
    wav_data = wav_data.astype(np.float64)

    f0, timeaxis = pw.harvest(
        wav_data, fs, f0_floor=71.0, f0_ceil=800, frame_period=5.0)
    sp = pw.cheaptrick(wav_data, f0, timeaxis, fs)
    mcep = pw.code_spectral_envelope(sp, fs, 36)

    # if you use sptk...
    # mcep = sptk.sp2mc(sp, order=36, alpha=0.42)

    return mcep


def JTES_analyze(in_dir, out_dir, emo_list):
    paths_emo = sorted(
        glob.glob('{}/{}/**'.format(in_dir, emo_list)))
    
    spk_idx = path_emo.split('/')[8]

    for path_emo in tqdm(paths_emo):
        os.makedirs(
            '{}/JTES_mcep/preprocess/{}/{}/npz'.format(out_dir, emo_list, spk_idx), exist_ok=True)
        os.makedirs(
            '{}/JTES_mcep/preprocess/{}/{}/npy'.format(out_dir, emo_list, spk_idx), exist_ok=True)

        paths_emo_wav = sorted(glob.glob(
            '{}/{}/{}/*.wav'.format(in_dir, emo_list, spk_idx)))

        mceps = []
        for path_emo_wav in paths_emo_wav:
            mcep_mean_and_std = pw_analyze(path_emo_wav)
            mceps.append(mcep_mean_and_std)

        mceps_concatenated = np.concatenate(mceps, axis=0)
        mceps_mean = np.mean(mceps_concatenated, axis=0, keepdims=False)
        mceps_std = np.std(mceps_concatenated, axis=0, keepdims=False)

        np.savez(
            '{}/JTES_mcep/preprocess/{}/{}/npz/{}.npz'.format(
                out_dir, emo_list, spk_idx, spk_idx),
            mceps_mean,
            mceps_std
        )

        for path_emo_wav in tqdm(paths_emo_wav):
            mcep = pw_analyze(path_emo_wav)
            normed_mcep = (mcep - mceps_mean) / mceps_std

            np.save(
                '{}/JTES_mcep/preprocess/{}/{}/npy/{}.npy'.format(
                    out_dir, emo_list, spk_idx, path_emo_wav.split('/')[-1].split('.')[0]),
                normed_mcep,
                allow_pickle=False
            )

if __name__ == "__main__":
    in_dir = '***'
    out_dir = '***'

    emo_lists = ['joy', 'sad', 'hap', 'neu']
    for emo_list in emo_lists:
        JTES_analyze(in_dir, out_dir, emo_list)