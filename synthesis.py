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
    text = np.array(text_to_sequence(text, hp.text_cleaners))
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
    num = 600000
    alpha = 1.0
    print("load model....")
    model = get_FastSpeech(num)
    wave_glow = utils.get_WaveGlow()
    tacotron2 = utils.get_Tacotron2()
    print("load model finish!")
    if not os.path.exists("results"):
        os.mkdir("results")

    texts = [
            "LEH1TS GOW1 AW2T TOW0 TH EH1RPAO2RT . TH PLEY1N LAE1NDIH0D TEH1N MIH1NAH0TS AH0GOW2 .",
            "AY1 LAH1V YUW1 VEH1RIY0 MAH1CH",
            "SAY1AH0NTIH0STS AE1T TH SER1N LAE1BRAH0TAO2RIY0 SEY1 DHEY1 HHAE1V DIH0SKAH1VER0D AH0 NUW1 PAA1RTAH0KAH0L .",
            "PREH1ZIH0DAH0NT TRAH1MP MEH1T WIH1TH AH1DHER0 LIY1DER0Z AE1T TH GRUW1P AH1V TWEH1NTIY0 KAA1NFER0AH0NS .",
            "VIH1PKIH0D IH0S AH0 CHAY0NIY1Z AO1NLAY2N EH2JHAH0KEY1SHAH0N FER1M DHAE1T AO1FER0Z AH0N AH0MEH1RAH0KAH0N EH2LAH0MEH1NER0IY0 EH2JHAH0KEY1SHAH0N IH0KSPIH1RIY0AH0NS TOW0 CHAY0NIY1Z STUW1DAH0NTS EY1JHD FAO1R TWEH1LV",
            "IH0N BIY1IH0NG KAH0MPEH1RAH0TIH0VLIY0 MAA1DER0N .",
            "AE1ND DIH0TEY1LIH0NG PAH0LIY1S IH0N SAH0VIH1LYAH0N KLOW1DHZ TOW0 B SKAE1TER0D THRUW0AW1T TH SAY1ZAH0BAH0L KRAW1D .",
            "PRIH1NTIH0NG , IH0N TH AO1NLIY0 SEH1NS WIH1TH HHWIH1CH W AA1R AE1T PRIY0ZEH1NT KAH0NSER1ND , DIH1FER0Z FRAH1M MOW2ST IH1F NAA1T FRAH1M AH0L TH AA1RTS AE1ND KRAE1FTS REH2PRIH0ZEH1NTIH0D IH0N TH EH2KSAH0BIH1SHAH0N",
            ]
    for words in texts:
        mel, mel_postnet, mel_torch, mel_postnet_torch = synthesis(
            model, words, alpha=alpha)

        Audio.tools.inv_mel_spec(mel_postnet, os.path.join(
            "results", words + "_" + str(num) + "_griffin_lim.wav"))

        waveglow.inference.inference(mel_postnet_torch, wave_glow, os.path.join(
            "results", words + "_" + str(num) + "_waveglow.wav"))

        mel_tac2, _, _, alignment = utils.load_data_from_tacotron2(words, tacotron2)
        waveglow.inference.inference(torch.stack([torch.from_numpy(
            mel_tac2).cuda()]), wave_glow, os.path.join("results", words + "_" + str(num) + "tacotron2.wav"))
        utils.plot_data([mel.numpy(), mel_postnet.numpy(), mel_tac2, alignment], words[:10])
        print("synthesis finish:", words)
