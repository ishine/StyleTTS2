# coding: utf-8
import os
import os.path as osp
import time
import random
import numpy as np
import random
import soundfile as sf
import librosa

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import pandas as pd

dicts = {" ": 0, "er": 1, "dui": 2, "lou": 3, "shi": 4, "cheng": 5, "jiao": 6, "yi": 7, "zhi": 8, "zuo": 9, "yong": 10,
         "zui": 11, "da": 12, "de": 13, "xian": 14, "ye": 15, "wei": 16, "di": 17, "fang": 18, "zheng": 19, "fu": 20,
         "yan": 21, "zhong": 22, "zi": 23, "liu": 24, "yue": 25, "hu": 26, "he": 27, "hao": 28, "te": 29, "shuai": 30,
         "xuan": 31, "bu": 32, "qu": 33, "xiao": 34, "gou": 35, "ge": 36, "bian": 37, "fen": 38, "jin": 39, "duo": 40,
         "jian": 41, "chu": 42, "le": 43, "bei": 44, "jing": 45, "shang": 46, "hai": 47, "guang": 48, "zhou": 49,
         "shen": 50, "zun": 51, "si": 52, "han": 53, "san": 54, "ya": 55, "huo": 56, "xiang": 57, "song": 58, "cai": 59,
         "rong": 60, "ce": 61, "sui": 62, "qi": 63, "hou": 64, "ji": 65, "qiang": 66, "yu": 67, "wang": 68, "xu": 69,
         "qiu": 70, "mi": 71, "qie": 72, "guan": 73, "dai": 74, "bao": 75, "gua": 76, "you": 77, "tao": 78, "zhu": 79,
         "bing": 80, "jie": 81, "qing": 82, "ying": 83, "kuan": 84, "gai": 85, "shan": 86, "ju": 87, "tiao": 88,
         "zai": 89, "ci": 90, "mai": 91, "pu": 92, "tong": 93, "pin": 94, "yin": 95, "hang": 96, "xing": 97, "shou": 98,
         "zhe": 99, "lai": 100, "jiu": 101, "ling": 102, "xin": 103, "chang": 104, "gong": 105, "ren": 106, "nian": 107,
         "ti": 108, "gao": 109, "cun": 110, "fei": 111, "chi": 112, "guo": 113, "tie": 114, "duan": 115, "gu": 116,
         "li": 117, "hui": 118, "zhai": 119, "zong": 120, "jia": 121, "ding": 122, "biao": 123, "zhun": 124, "wai": 125,
         "ze": 126, "dang": 127, "shu": 128, "yang": 129, "nei": 130, "chan": 131, "bi": 132, "jiang": 133, "xi": 134,
         "kuai": 135, "ku": 136, "hua": 137, "yao": 138, "xun": 139, "su": 140, "she": 141, "pei": 142, "kong": 143,
         "cha": 144, "bie": 145, "deng": 146, "mian": 147, "min": 148, "ping": 149, "wen": 150, "fa": 151, "ni": 152,
         "hong": 153, "ri": 154, "gui": 155, "lv": 156, "chao": 157, "bai": 158, "lian": 159, "kan": 160, "fan": 161,
         "tai": 162, "tou": 163, "tian": 164, "xia": 165, "quan": 166, "ben": 167, "zeng": 168, "zhang": 169,
         "kua": 170, "e": 171, "qian": 172, "jun": 173, "ao": 174, "mo": 175, "liang": 176, "wu": 177, "mu": 178,
         "dao": 179, "yuan": 180, "ban": 181, "po": 182, "dong": 183, "chong": 184, "run": 185, "pi": 186, "diao": 187,
         "wan": 188, "kao": 189, "zhao": 190, "ke": 191, "tuan": 192, "ta": 193, "dan": 194, "mei": 195, "suo": 196,
         "cuo": 197, "ceng": 198, "ru": 199, "neng": 200, "cong": 201, "que": 202, "kuo": 203, "du": 204, "long": 205,
         "dou": 206, "nan": 207, "hen": 208, "shuo": 209, "ran": 210, "geng": 211, "xie": 212, "men": 213, "ming": 214,
         "xue": 215, "xiu": 216, "can": 217, "lu": 218, "yun": 219, "rang": 220, "chuang": 221, "zao": 222, "tui": 223,
         "sheng": 224, "dian": 225, "gei": 226, "kai": 227, "za": 228, "suan": 229, "zhan": 230, "chuan": 231,
         "ting": 232, "wo": 233, "liao": 234, "cu": 235, "tu": 236, "rui": 237, "gang": 238, "zhuan": 239, "lun": 240,
         "na": 241, "luo": 242, "weng": 243, "zhuo": 244, "lei": 245, "re": 246, "ka": 247, "huan": 248, "zou": 249,
         "bang": 250, "mie": 251, "qin": 252, "pan": 253, "zhua": 254, "shei": 255, "pai": 256, "lan": 257, "nao": 258,
         "xiong": 259, "chou": 260, "ba": 261, "shao": 262, "rao": 263, "mao": 264, "qiao": 265, "shui": 266,
         "lie": 267, "a": 268, "zen": 269, "me": 270, "feng": 271, "kou": 272, "fou": 273, "shun": 274, "sai": 275,
         "zu": 276, "nv": 277, "sou": 278, "tan": 279, "kun": 280, "sha": 281, "man": 282, "huang": 283, "gan": 284,
         "rou": 285, "ai": 286, "nong": 287, "tuo": 288, "che": 289, "lin": 290, "pian": 291, "kang": 292, "ma": 293,
         "cui": 294, "zhuang": 295, "an": 296, "chen": 297, "zhui": 298, "shua": 299, "peng": 300, "sang": 301,
         "ning": 302, "gen": 303, "niu": 304, "nai": 305, "huai": 306, "guai": 307, "zhen": 308, "mou": 309, "lao": 310,
         "qun": 311, "reng": 312, "jue": 313, "lve": 314, "pang": 315, "shuang": 316, "kuang": 317, "meng": 318,
         "cao": 319, "teng": 320, "chai": 321, "ken": 322, "chun": 323, "chui": 324, "bo": 325, "ou": 326, "lang": 327,
         "miao": 328, "zuan": 329, "zha": 330, "nu": 331, "sao": 332, "mang": 333, "la": 334, "shai": 335, "sen": 336,
         "hun": 337, "gun": 338, "tang": 339, "die": 340, "leng": 341, "heng": 342, "se": 343, "pao": 344, "piao": 345,
         "pen": 346, "sun": 347, "zei": 348, "ha": 349, "bin": 350, "diu": 351, "niang": 352, "juan": 353, "hei": 354,
         "wa": 355, "pa": 356, "nuo": 357, "ga": 358, "ruo": 359, "ca": 360, "dun": 361, "ruan": 362, "luan": 363,
         "zang": 364, "fo": 365, "qiong": 366, "niao": 367, "kui": 368, "nve": 369, "sa": 370, "qia": 371, "dei": 372,
         "lia": 373, "cang": 374, "tun": 375, "pie": 376, "en": 377, "shuan": 378, "zan": 379, "nin": 380, "nie": 381,
         "ang": 382, "keng": 383, "beng": 384, "nang": 385, "cou": 386, "cuan": 387, "yai": 388, "nuan": 389,
         "pou": 390, "chuai": 391, "cen": 392, "chuo": 393, "nen": 394, "ne": 395, "zhuai": 396, "seng": 397,
         "jiong": 398, "miu": 399, "lo": 400, "o": 401, }


class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts

    def __call__(self, text):
        indexes = []
        chars = [i for j in text.split() for i in (j, ' ')][:-1]
        # print(text)
        for char in chars:

            if char not in self.word_index_dictionary:
                #print(text)
                #print(char)
                #quit()
                continue
            else:
                indexes.append(self.word_index_dictionary[char])

        return indexes


np.random.seed(1)
random.seed(1)
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
}

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 root_path,
                 sr=24000,
                 data_augmentation=False,
                 validation=False,
                 OOD_data="PreProcess/ood_text_neutral.txt",
                 min_length=50,
                 ):

        spect_params = SPECT_PARAMS
        mel_params = MEL_PARAMS

        #_data_list = [l.strip().split('|') for l in data_list]
        #DO NOT STRIP
        _data_list = [l.replace('\n', "").split('|') for l in data_list]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner()
        self.sr = sr

        self.df = pd.DataFrame(self.data_list)

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192

        self.min_length = min_length
        with open(OOD_data, 'r', encoding='utf-8') as f:
            tl = f.readlines()
        idx = 1 if '.wav' in tl[0].split('|')[0] else 0
        self.ptexts = [t.split('|')[idx] for t in tl]

        self.root_path = root_path

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        path = data[0]

        wave, text_tensor, speaker_id = self._load_tensor(data)

        mel_tensor = preprocess(wave).squeeze()

        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]

        #print("Id", speaker_id)
        #print(self.df)
        #print(self.df[self.df[2] == str(speaker_id)+"\n"])

        # get reference sample
        ref_data = (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
        ref_mel_tensor, ref_label = self._load_data(ref_data[:3])

        # get OOD text

        ps = ""

        while len(ps) < self.min_length:

            # this is modified, I think it's weird?

            if len(self.ptexts) - 1 <= 0:
                a = 1
                #print("This is somewhat problematic")

            rand_idx = np.random.randint(0, len(self.ptexts))
            ps = self.ptexts[rand_idx]

            text = self.text_cleaner(ps)
            text.insert(0, 0)
            text.append(0)

            ref_text = torch.LongTensor(text)

        return speaker_id, acoustic_feature, text_tensor, ref_text, ref_mel_tensor, ref_label, path, wave

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        wave, sr = sf.read(osp.join(self.root_path, wave_path))
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
            print(wave_path, sr)

        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)

        text = self.text_cleaner(text)

        text.insert(0, 0)
        text.append(0)

        text = torch.LongTensor(text)

        return wave, text, speaker_id

    def _load_data(self, data):
        wave, text_tensor, speaker_id = self._load_tensor(data)
        mel_tensor = preprocess(wave).squeeze()

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, speaker_id


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])
        max_rtext_length = max([b[3].shape[0] for b in batch])

        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        ref_texts = torch.zeros((batch_size, max_rtext_length)).long()

        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        paths = ['' for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]

        for bid, (label, mel, text, ref_text, ref_mel, ref_label, path, wave) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            rtext_size = ref_text.size(0)
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel

            ref_labels[bid] = ref_label
            waves[bid] = wave

        return waves, texts, input_lengths, ref_texts, ref_lengths, mels, output_lengths, ref_mels


def build_dataloader(path_list,
                     root_path,
                     validation=False,
                     OOD_data="PreProcess/ood_text_neutral.txt",
                     min_length=50,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):
    dataset = FilePathDataset(path_list, root_path, OOD_data=OOD_data, min_length=min_length, validation=validation,
                              **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader

# %%
