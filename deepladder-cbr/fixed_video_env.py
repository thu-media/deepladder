import os
import numpy as np
import h5py

encode_ladder = [100, 300, 750, 1250, 1850, 2850, 4300, 5700, 7000]
encode_h = [144, 240, 360, 480, 720, 1080]
INFO = [224, 224, 3]
IMAGE_COUNT = 4
RANDOM_SEED = 42


class VideoEnv:
    def read_element(self, filename):
        f = open(filename, 'r')
        output_arr = []
        while True:
            res_arr = []
            for _ in range(len(encode_h)):
                arr = []
                line = f.readline()
                line = line.strip()
                if not line:
                    f.close()
                    return output_arr
                _sp = line.split(',')
                for _idx in _sp:
                    arr.append(float(_idx))
                res_arr.append(arr)
            output_arr.append(res_arr)

    def read_features(self, filename):
        f = h5py.File(filename,'r') 
        # [4, 2048]
        arr = np.array(f['data'])
        f.close()
        return arr


    def __init__(self, random_seed=RANDOM_SEED):
        self.video_size = []
        self.video_vmaf = []
        self.video_features = []
        self.video_count = 0
        self.video_name = []
        for p in os.listdir('./test_size/'):
            # self.video_name.append(p)
            for size_ in self.read_element('./test_size/' + p):
                self.video_size.append(size_)
                self.video_name.append(p)
            for vmaf_ in self.read_element('./test_vmaf/' + p):
                self.video_vmaf.append(vmaf_)
            for feature_ in self.read_features('./test_feature/' + p + '.h5'):
                self.video_features.append(feature_)
        self.video_count = len(self.video_size)
        self.video_idx = 0

    def get_features(self):
        self.video_idx += 1  #np.random.randint(self.video_count)
        self.video_idx %= self.video_count
        feat = self.video_features[self.video_idx]
        return feat
        
    def get_video_name(self):
        return self.video_name[self.video_idx]

    def step(self, input_bitrate_ladder):
        # assert len(bitrate_ladder) == len(encode_h)
        vmaf_arr, size_arr = [], []
        bitrate_ladder = np.array(input_bitrate_ladder) * 1000. * 6.
        for resolution, bitrate in enumerate(bitrate_ladder):
            # bitrate = norm_bitrate * 7000.
            if bitrate > 0:
                vmaf_curve = self.video_vmaf[self.video_idx][resolution]
                size_curve = self.video_size[self.video_idx][resolution]
                _bit_index = len(encode_ladder) - 1
                for idx, p in enumerate(encode_ladder):
                    if p >= bitrate:
                        _bit_index = idx
                        break
                if _bit_index == 0:
                    bit_range = [0, encode_ladder[_bit_index]]
                    vmaf_tmp = [0, vmaf_curve[_bit_index]]
                    size_tmp = [0, size_curve[_bit_index]]
                # elif _bit_index == len(encode_ladder) - 1:
                #     bit_range = [encode_ladder[_bit_index], 10000.]
                #     vmaf_tmp = [vmaf_curve[_bit_index], 100.]
                #     size_tmp = [size_curve[_bit_index], size_curve[_bit_index] / encode_ladder[-1] * 10000.]
                else:
                    bit_range = [encode_ladder[_bit_index - 1], encode_ladder[_bit_index]]
                    vmaf_tmp = [vmaf_curve[_bit_index - 1], vmaf_curve[_bit_index]]
                    size_tmp = [size_curve[_bit_index - 1], size_curve[_bit_index]]
                # print(vmaf_curve, vmaf_tmp)
                est_vmaf = vmaf_tmp[0] + (vmaf_tmp[1] - vmaf_tmp[0]) / \
                    (bit_range[1] - bit_range[0]) * (bitrate - bit_range[0])
                est_size = size_tmp[0] + (size_tmp[1] - size_tmp[0]) / \
                    (bit_range[1] - bit_range[0]) * (bitrate - bit_range[0])
                assert est_vmaf >= 0.
                assert est_vmaf <= 100.
                vmaf_arr.append(est_vmaf)
                size_arr.append(est_size)
            else:
                vmaf_arr.append(0.)
                size_arr.append(0.)

        return np.array(vmaf_arr), np.array(size_arr)

if __name__ == '__main__':
    pass
