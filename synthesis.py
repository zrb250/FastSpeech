import torch
import torch.nn as nn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from fastspeech import FastSpeech
from text import text_to_sequence
import hparams as hp
import utils
import Audio
import glow
import waveglow

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_FastSpeech(num):
    checkpoint_path = "checkpoint_" + str(num) + ".pth.tar"
    model = nn.DataParallel(FastSpeech()).to(device)
    model.load_state_dict(torch.load(os.path.join(
        hp.checkpoint_path, checkpoint_path))['model'])
    model.eval()

    return model


def synthesis(model, text, alpha=1.0):
    print("text:", text)
    text = np.array(text_to_sequence(text, hp.text_cleaners))
    print("textid:", text)
    text = np.stack([text])

    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    with torch.no_grad():
        sequence = torch.autograd.Variable(
            torch.from_numpy(text)).cuda().long()
        src_pos = torch.autograd.Variable(
            torch.from_numpy(src_pos)).cuda().long()

        mel, mel_postnet = model.module.forward(sequence, src_pos, alpha=alpha)

        return mel[0].cpu().transpose(0, 1), \
            mel_postnet[0].cpu().transpose(0, 1), \
            mel.transpose(1, 2), \
            mel_postnet.transpose(1, 2)


if __name__ == "__main__":
    # Test
    num = 350000
    alpha = 1.0
    model = get_FastSpeech(num)
    #words = "Let’s go out to the airport. The plane landed ten minutes ago."
    words = "IH1_B N_E B_B IY1_I IH0_I NG_E K_B AH0_I M_I P_I EH1_I R_I AH0_I T_I IH0_I V_I L_I IY0_E M_B AA1_I D_I ER0_I N_E"
    #words = "L_B EH1_I T_I S_E G_B OW1_E AW2_B T_E T_B OW0_E TH_S EH1_B R_I P_I AO2_I R_I T_E SIL TH_S P_B L_I EY1_I N_E L_B AE1_I N_I D_I IH0_I D_E T_B EH1_I N_E M_B IH1_I N_I AH0_I T_I S_E AH0_B G_I OW2_E SIL" 

    mel, mel_postnet, mel_torch, mel_postnet_torch = synthesis(
        model, words, alpha=alpha)

    if not os.path.exists("results"):
        os.mkdir("results")
    Audio.tools.inv_mel_spec(mel_postnet, os.path.join(
        "results", words + "_" + str(num) + "_griffin_lim.wav"))

    wave_glow = utils.get_WaveGlow()
    waveglow.inference.inference(mel_postnet_torch, wave_glow, os.path.join(
        "results", words + "_" + str(num) + "_waveglow.wav"))

    tacotron2 = utils.get_Tacotron2()
    mel_tac2, _, _ = utils.load_data_from_tacotron2(words, tacotron2)
    waveglow.inference.inference(torch.stack([torch.from_numpy(
        mel_tac2).cuda()]), wave_glow, os.path.join("results", "tacotron2.wav"))

    print("synthesis completed")
    utils.plot_data([mel.numpy(), mel_postnet.numpy(), mel_tac2])
