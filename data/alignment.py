import numpy as np
import os
import utils

target_frame_t = 0.04644
target_frame_shift_t = 0.01161

src_frame_t = 0.025
src_frame_shift_t = 0.010


def train_txt():

    out_dir = './'
    index = 1
    phone_align_file = './to16kList.txt.out'
    f=open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8')
    funiq=open(os.path.join(out_dir, 'train_uniq.txt'), 'w', encoding='utf-8')
    for line in open(phone_align_file):
        wav, txt, phones, _, alignInfo, plen = line.strip().split("$$")

        ph = [x.split('_')[0] for x in phones.split(' ')]
        f.write(phones + '\n')
        funiq.write(' '.join(ph) + '\n')
        index = index + 1
        if(index % 100 == 0):
            print("complete:", index);

    return;

def alignment():

    in_dir = '../mels'
    out_dir = '../alignments-self'
    index = 1
    # phone_align_file = './test.txt'
    phone_align_file = './to16kList.txt.out'
    targetTimeSpan = get_target_time_list();
    for line in open(phone_align_file):
        wav, txt, phones, _, alignInfo, plen = line.strip().split("$$")
        # print(phones)
        # print(alignInfo)

        mel_filename = 'ljspeech-mel-%05d.npy' % index
        mel = np.load(os.path.join(in_dir, mel_filename))
        tlen = mel.shape[0]
        tlen = tlen - 4; # del padding size
        _, D, _ = get_new_align(alignInfo, targetTimeSpan, tlen);

        # break;
        np.save(os.path.join(out_dir, str(index - 1) + ".npy"), D, allow_pickle=False)
        index = index + 1

        if(index % 100 == 0):
            print("complete:", index);

    return;

def get_new_align(alignInfo, targetTimeSpan, tlen):
    alignlist = np.array(alignInfo.split(' ')).astype(np.int)
    timeSpan= []
    laststep = 0.0
    for i in range(len(alignlist)):
        frame_n = alignlist[i]
        start, end = get_frame_src_time_span(laststep, frame_n)
        laststep = end;
        timeSpan.append([start, end])

    Dlen = len(timeSpan);
    D = np.array([0 for _ in range(Dlen)])

    idx= 0;
    cur = timeSpan[idx];
    next = timeSpan[idx + 1];
    for tidx in range(tlen):
        tframe = targetTimeSpan[tidx]
        if(idx < Dlen):
            if is_cur(cur, next, tframe) == True:
                D[idx] = D[idx] + 1
            elif(idx < (Dlen - 1)):
                D[idx + 1] = D[idx + 1] + 1

                #tstart > cend, goto next phone
                if(tframe[0] > cur[1]):
                    idx= idx + 1;
                    cur = timeSpan[idx];
                    if(idx < (Dlen - 1)):
                        next = timeSpan[idx + 1];
            else:
                D[idx] = D[idx] + 1
                # print("append frame:", tidx, targetTimeSpan[tidx])

    # for i in range(len(alignlist)):
    #     print(D[i], alignlist[i])
    #add padding frame to start and end
    D[0] = D[0] + 2;
    D[len(D) - 1] = D[len(D) - 1] + 2

    # print(len(alignlist),alignInfo)
    # dl=D.astype(str).tolist();
    # print(len(D)," ".join(dl))
    # print("   ")
    return  alignInfo, D, len(D)

def is_cur(cur, next, tframe):
    cstart, cend = cur;
    nstart, nend = next;
    tstart, tend = tframe;
    # print("cur, next, tframe", cur, next, tframe)
    # if(tstart >= nend):
    #     print("un know!!, cur, next, tframe", cur, next, tframe)
    #     exit(0)
    if(tend <= cend):
        return True;
    if((cend - tstart) > (tend - nstart)):
        return True;
    else:
        return False;

def get_target_time_list():
    laststep = 0.0
    targetTimeSpan = []
    for i in range(1200):
        start, end = get_frame_target_time_span(laststep, 1)
        laststep = end;
        targetTimeSpan.append([start, end])
    return targetTimeSpan

def get_frame_src_time_span(laststep, frame_n):

    timespan = (frame_n - 1) * src_frame_shift_t + src_frame_t;
    start = 0.0
    if(laststep > 0):
        start = laststep - (src_frame_t - src_frame_shift_t);
    end = start + timespan

    return round(start,5), round(end,5)

def get_frame_target_time_span(laststep, frame_n):
    timespan = (frame_n - 1) * target_frame_shift_t + target_frame_t;
    start = 0.0
    if(laststep > 0):
        start = laststep - (target_frame_t - target_frame_shift_t);
    end = start + timespan

    return round(start,5), round(end,5)

def plot_alignment():
    in_dir = '../self_alignment/alignments'
    t2_in_dir = '../alignments'

    for num in range(20):
        fname = str(num) + ".npy";
        a = np.load(os.path.join(in_dir, fname))
        align = np.zeros((a.shape[0],a.sum()), dtype=float)

        last = 0;
        for i in range(len(a)):
            num = a[i];
            for j in range(num):
              pos = last + j;
              align[i][pos] = 1.0
            last = num + last;


        t2a = np.load(os.path.join(t2_in_dir, fname))
        t2align = np.zeros((t2a.shape[0],t2a.sum()), dtype=float)
        last = 0;
        for i in range(len(t2a)):
            num = t2a[i];
            for j in range(num):
                pos = last + j;
                t2align[i][pos] = 1.0
            last = num + last;

        utils.plot_data([align, t2align], fname)


if __name__ == '__main__':
    # alignment();
    # train_txt()
    plot_alignment();