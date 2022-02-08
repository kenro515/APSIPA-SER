import glob
import torch
import numpy as np
import torch.utils.data as data
from torchvision import transforms


def prepare_leave_one_group_out(p_paths):
    spk_idx_list = [
        ["m01", "m11", "m21", "m31", "m41", "f01", "f11", "f21", "f31", "f41",
            "m02", "m12", "m22", "m32", "m42", "f02", "f12", "f22", "f32", "f42"],
        ["m03", "m13", "m23", "m33", "m43", "f03", "f13", "f23", "f33", "f43",
            "m04", "m14", "m24", "m34", "m44", "f04", "f14", "f24", "f34", "f44"],
        ["m05", "m15", "m25", "m35", "m45", "f05", "f15", "f25", "f35", "f45",
            "m06", "m16", "m26", "m36", "m46", "f06", "f16", "f26", "f36", "f46"],
        ["m07", "m17", "m27", "m37", "m47", "f07", "f17", "f27", "f37", "f47",
            "m08", "m18", "m28", "m38", "m48", "f08", "f18", "f28", "f38", "f48"],
        ["m09", "m19", "m29", "m39", "m49", "f09", "f19", "f29", "f39", "f49",
            "m10", "m20", "m30", "m40", "m50", "f10", "f20", "f30", "f40", "f50"]
    ]

    seg_fold_list_0 = []
    seg_fold_list_1 = []
    seg_fold_list_2 = []
    seg_fold_list_3 = []
    seg_fold_list_4 = []

    for path in p_paths:
        if path.split('/')[-1].split('_')[0] in spk_idx_list[0]:
            seg_fold_list_0.append(path)
        elif path.split('/')[-1].split('_')[0] in spk_idx_list[1]:
            seg_fold_list_1.append(path)
        elif path.split('/')[-1].split('_')[0] in spk_idx_list[2]:
            seg_fold_list_2.append(path)
        elif path.split('/')[-1].split('_')[0] in spk_idx_list[3]:
            seg_fold_list_3.append(path)
        else:
            seg_fold_list_4.append(path)

    print("fold0:{}\tfold1:{}\tfold2:{}\tfold3:{}\tfold4:{}".format(
        len(seg_fold_list_0),
        len(seg_fold_list_1),
        len(seg_fold_list_2),
        len(seg_fold_list_3),
        len(seg_fold_list_4)))

    seg_fold_list_out = seg_fold_list_0 + seg_fold_list_1 + \
        seg_fold_list_2 + seg_fold_list_3 + seg_fold_list_4

    return seg_fold_list_out


class MyDataset(data.Dataset):
    def __init__(self, dir_name, in_dim=None, max_len=None):
        super(MyDataset, self).__init__()

        self.dir_name = dir_name

        self.sound_paths = prepare_leave_one_group_out(sorted(
            glob.glob(self.dir_name)))

        self.len = len(self.sound_paths)
        self.in_dim = in_dim
        self.max_len = max_len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        p = self.sound_paths[index]
        sound_data = np.load(p)

        # triming and padding
        if sound_data.shape[0] > self.max_len:
            sound_data = sound_data[0:self.max_len, :]
        else:
            padding_size = self.max_len-sound_data.shape[0]
            buff = np.zeros([padding_size, self.in_dim])

            sound_data = np.concatenate([sound_data, buff])

        sound_data = np.transpose(sound_data, (1, 0))
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        sound_data = transform(sound_data)

        # ** check! ** make categorical emotion label
        emo_list = p.split('/')[6]
        emo_label = np.zeros(4, dtype='int64')
        if emo_list == "ang":
            emo_label[0] = 1
        elif emo_list == "joy":
            emo_label[1] = 1
        elif emo_list == "sad":
            emo_label[2] = 1
        else:
            emo_label[3] = 1

        return sound_data.type(torch.cuda.FloatTensor), torch.from_numpy(emo_label)
