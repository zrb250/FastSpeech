import numpy as np
import os

target_frame_t = 0.046439
target_frame_shift_t = 0.011609

src_frame_t = 0.025
src_frame_shift_t = 0.010


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
                idx= idx + 1;
                cur = next;
                next = timeSpan[idx];
            else:
                D[idx] = D[idx] + 1
                # print("append frame:", tidx, targetTimeSpan[tidx])

    #add padding frame to start and end
    D[0] = D[0] + 2;
    D[len(D) - 1] = D[len(D) - 1] + 2
    # print(D, D.sum(), len(D))
    return  alignInfo, D, len(D)

def is_cur(cur, next, tframe):
    cstart, cend = cur;
    nstart, nend = next;
    tstart, tend = tframe;
    if(tend <= cend):
        return True;
    if(tstart >= cend):
        return False;
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

    return round(start,6), round(end,6)

def get_frame_target_time_span(laststep, frame_n):
    timespan = (frame_n - 1) * target_frame_shift_t + target_frame_t;
    start = 0.0
    if(laststep > 0):
        start = laststep - (target_frame_t - target_frame_shift_t);
    end = start + timespan

    return round(start,6), round(end,6)

if __name__ == '__main__':
    alignment();