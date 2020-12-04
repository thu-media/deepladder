import os
import numpy as np
import h5py

vbr_ladder = np.arange(20, 40)
# encode_w = []
encode_h = [144, 240, 360, 480, 720, 1080]
INFO = [224, 224, 3]
IMAGE_COUNT = 4
RANDOM_SEED = 42

# ffmpeg_quality_metrics test/45-x265.mp4 test1.mp4 --enable-vmaf -of csv >> quality3.csv

class VideoEnv:
    def read_elements(self, filename):
        f = open(filename, 'r')
        arr = []
        for line in f:
            tmp = []
            sp = line.split(',')
            for p in sp:
                tmp.append(float(p))
            arr.append(np.array(tmp))
        return np.array(arr)

    def read_features(self, filename):
        f = h5py.File(filename, 'r')
        # [4, 2048]
        arr = np.array(f['data'])
        f.close()
        return arr

    def __init__(self, random_seed=RANDOM_SEED):
        np.random.seed(random_seed)
        self.video_name = []
        self.video_ssim = []
        self.video_size = []
        # self.video_name = []
        for p in os.listdir('./test/ssim/'):
            self.video_name.append(p)
            self.video_ssim.append(self.read_elements('./test/ssim/' + p))
            self.video_size.append(self.read_elements('./test/size/' + p))
        self.video_count = len(self.video_name)
        self.video_idx = 0

    def get_video_name(self):
        return self.video_name[self.video_idx]
        
    def get_features(self, idx = None):
        if idx is None:
            self.video_idx = np.random.randint(self.video_count)
        else:
            self.video_idx = idx % self.video_count
        feat = self.read_features('./test/feature/' + self.video_name[self.video_idx] + '.h5')
        return feat

    def step(self, input_vbr_ladder):
        # assert len(bitrate_ladder) == len(encode_h)
        vmaf_arr, size_arr = [], []
        for resolution, _vbr in enumerate(input_vbr_ladder):
            _ssim = self.video_ssim[self.video_idx][resolution][_vbr]
            _size = self.video_size[self.video_idx][resolution][_vbr]
            # _vqa = 2062.3 * \
            #     (1./(1. + np.exp(-11.8 * (_ssim - 1.3)))+0.5) + \
            #     40.6 * _ssim - 1035.6
            _vqa = 3.27 * (0.5 - 1. / (1 + np.exp(37.37*(_ssim - 0.93)))) + 1.22 * _ssim + 2.14
            _vqa *= 20.
            vmaf_arr.append(_vqa)
            size_arr.append(_size)
                
        return np.array(vmaf_arr), np.array(size_arr)


if __name__ == '__main__':
    pass
