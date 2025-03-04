""" from https://github.com/keithito/tacotron """

import re


# valid_symbols = [
#     'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
#     'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
#     'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
#     'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
#     'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
#     'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
#     'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
# ]

valid_symbols = ['<eps>', 'SIL', 'SIL_B', 'SIL_E', 'SIL_I', 'SIL_S', 'SPN', 'SPN_B', 'SPN_E',
                 'SPN_I', 'SPN_S', 'AA_B', 'AA_E', 'AA_I', 'AA_S', 'AA0_B', 'AA0_E', 'AA0_I',
                 'AA0_S', 'AA1_B', 'AA1_E', 'AA1_I', 'AA1_S', 'AA2_B', 'AA2_E', 'AA2_I', 'AA2_S',
                 'AE_B', 'AE_E', 'AE_I', 'AE_S', 'AE0_B', 'AE0_E', 'AE0_I', 'AE0_S', 'AE1_B', 'AE1_E',
                 'AE1_I', 'AE1_S', 'AE2_B', 'AE2_E', 'AE2_I', 'AE2_S', 'AH_B', 'AH_E', 'AH_I', 'AH_S',
                 'AH0_B', 'AH0_E', 'AH0_I', 'AH0_S', 'AH1_B', 'AH1_E', 'AH1_I', 'AH1_S', 'AH2_B', 'AH2_E',
                 'AH2_I', 'AH2_S', 'AO_B', 'AO_E', 'AO_I', 'AO_S', 'AO0_B', 'AO0_E', 'AO0_I', 'AO0_S',
                 'AO1_B', 'AO1_E', 'AO1_I', 'AO1_S', 'AO2_B', 'AO2_E', 'AO2_I', 'AO2_S', 'AW_B', 'AW_E',
                 'AW_I', 'AW_S', 'AW0_B', 'AW0_E', 'AW0_I', 'AW0_S', 'AW1_B', 'AW1_E', 'AW1_I', 'AW1_S',
                 'AW2_B', 'AW2_E', 'AW2_I', 'AW2_S', 'AY_B', 'AY_E', 'AY_I', 'AY_S', 'AY0_B', 'AY0_E',
                 'AY0_I', 'AY0_S', 'AY1_B', 'AY1_E', 'AY1_I', 'AY1_S', 'AY2_B', 'AY2_E', 'AY2_I', 'AY2_S',
                 'B_B', 'B_E', 'B_I', 'B_S', 'CH_B', 'CH_E', 'CH_I', 'CH_S', 'D_B', 'D_E', 'D_I', 'D_S',
                 'DH_B', 'DH_E', 'DH_I', 'DH_S', 'EH_B', 'EH_E', 'EH_I', 'EH_S', 'EH0_B', 'EH0_E', 'EH0_I',
                 'EH0_S', 'EH1_B', 'EH1_E', 'EH1_I', 'EH1_S', 'EH2_B', 'EH2_E', 'EH2_I', 'EH2_S',
                 'ER_B', 'ER_E', 'ER_I', 'ER_S', 'ER0_B', 'ER0_E', 'ER0_I', 'ER0_S', 'ER1_B', 'ER1_E',
                 'ER1_I', 'ER1_S', 'ER2_B', 'ER2_E', 'ER2_I', 'ER2_S', 'EY_B', 'EY_E', 'EY_I', 'EY_S',
                 'EY0_B', 'EY0_E', 'EY0_I', 'EY0_S', 'EY1_B', 'EY1_E', 'EY1_I', 'EY1_S', 'EY2_B', 'EY2_E',
                 'EY2_I', 'EY2_S', 'F_B', 'F_E', 'F_I', 'F_S', 'G_B', 'G_E', 'G_I', 'G_S', 'HH_B', 'HH_E',
                 'HH_I', 'HH_S', 'IH_B', 'IH_E', 'IH_I', 'IH_S', 'IH0_B', 'IH0_E', 'IH0_I', 'IH0_S', 'IH1_B',
                 'IH1_E', 'IH1_I', 'IH1_S', 'IH2_B', 'IH2_E', 'IH2_I', 'IH2_S', 'IY_B', 'IY_E', 'IY_I', 'IY_S',
                 'IY0_B', 'IY0_E', 'IY0_I', 'IY0_S', 'IY1_B', 'IY1_E', 'IY1_I', 'IY1_S', 'IY2_B', 'IY2_E', 'IY2_I',
                 'IY2_S', 'JH_B', 'JH_E', 'JH_I', 'JH_S', 'K_B', 'K_E', 'K_I', 'K_S', 'L_B', 'L_E', 'L_I', 'L_S',
                 'M_B', 'M_E', 'M_I', 'M_S', 'N_B', 'N_E', 'N_I', 'N_S', 'NG_B', 'NG_E', 'NG_I', 'NG_S', 'OW_B',
                 'OW_E', 'OW_I', 'OW_S', 'OW0_B', 'OW0_E', 'OW0_I', 'OW0_S', 'OW1_B', 'OW1_E', 'OW1_I', 'OW1_S',
                 'OW2_B', 'OW2_E', 'OW2_I', 'OW2_S', 'OY_B', 'OY_E', 'OY_I', 'OY_S', 'OY0_B', 'OY0_E', 'OY0_I',
                 'OY0_S', 'OY1_B', 'OY1_E', 'OY1_I', 'OY1_S', 'OY2_B', 'OY2_E', 'OY2_I', 'OY2_S', 'P_B', 'P_E',
                 'P_I', 'P_S', 'R_B', 'R_E', 'R_I', 'R_S', 'S_B', 'S_E', 'S_I', 'S_S', 'SH_B', 'SH_E', 'SH_I',
                 'SH_S', 'T_B', 'T_E', 'T_I', 'T_S', 'TH_B', 'TH_E', 'TH_I', 'TH_S', 'UH_B', 'UH_E', 'UH_I', 'UH_S',
                 'UH0_B', 'UH0_E', 'UH0_I', 'UH0_S', 'UH1_B', 'UH1_E', 'UH1_I', 'UH1_S', 'UH2_B', 'UH2_E', 'UH2_I',
                 'UH2_S', 'UW_B', 'UW_E', 'UW_I', 'UW_S', 'UW0_B', 'UW0_E', 'UW0_I', 'UW0_S', 'UW1_B', 'UW1_E',
                 'UW1_I', 'UW1_S', 'UW2_B', 'UW2_E', 'UW2_I', 'UW2_S', 'V_B', 'V_E', 'V_I', 'V_S', 'W_B',
                 'W_E', 'W_I', 'W_S', 'Y_B', 'Y_E', 'Y_I', 'Y_S', 'Z_B', 'Z_E', 'Z_I', 'Z_S', 'ZH_B', 'ZH_E',
                 'ZH_I', 'ZH_S', '#0', '#1', '#2', '#3', '#4', '#5', '#6', '#7', '#8', '#9', '#10', '#11',
                 '#12', '#13', '#14', '#15', '#16'
                 ]


_valid_symbol_set = set(valid_symbols)


class CMUDict:
    '''Thin wrapper around CMUDict data. http://www.speech.cs.cmu.edu/cgi-bin/cmudict'''

    def __init__(self, file_or_path, keep_ambiguous=True):
        if isinstance(file_or_path, str):
            with open(file_or_path, encoding='latin-1') as f:
                entries = _parse_cmudict(f)
        else:
            entries = _parse_cmudict(file_or_path)
        if not keep_ambiguous:
            entries = {word: pron for word,
                       pron in entries.items() if len(pron) == 1}
        self._entries = entries

    def __len__(self):
        return len(self._entries)

    def lookup(self, word):
        '''Returns list of ARPAbet pronunciations of the given word.'''
        return self._entries.get(word.upper())


_alt_re = re.compile(r'\([0-9]+\)')


def _parse_cmudict(file):
    cmudict = {}
    for line in file:
        if len(line) and (line[0] >= 'A' and line[0] <= 'Z' or line[0] == "'"):
            parts = line.split('  ')
            word = re.sub(_alt_re, '', parts[0])
            pronunciation = _get_pronunciation(parts[1])
            if pronunciation:
                if word in cmudict:
                    cmudict[word].append(pronunciation)
                else:
                    cmudict[word] = [pronunciation]
    return cmudict


def _get_pronunciation(s):
    parts = s.strip().split(' ')
    for part in parts:
        if part not in _valid_symbol_set:
            return None
    return ' '.join(parts)
