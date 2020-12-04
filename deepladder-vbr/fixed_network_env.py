import os
import numpy as np
import fixed_vbr_env

COOKED_TRACE_FOLDER = './hd-fs-test/'
S_INFO = 4
S_LEN = 15
A_DIM = 6
DURATION = 4
RANDOM_SEED = 42

class FixedNetworkEnvDiscrete:
    def load_trace(self, cooked_trace_folder):
        cooked_files = os.listdir(cooked_trace_folder)
        all_cooked_time = []
        all_cooked_bw = []
        all_file_names = []
        for cooked_file in cooked_files:
            file_path = cooked_trace_folder + cooked_file
            cooked_time = []
            cooked_bw = []
            # print file_path
            with open(file_path, 'rb') as f:
                for line in f:
                    parse = line.split()
                    cooked_time.append(float(parse[0]))
                    cooked_bw.append(float(parse[1]) * 0.95)
            all_cooked_time.append(cooked_time)
            all_cooked_bw.append(cooked_bw)
            all_file_names.append(cooked_file)

        return all_cooked_time, all_cooked_bw, all_file_names

    def __init__(self, cooked_trace_folder=COOKED_TRACE_FOLDER, random_seed = RANDOM_SEED, a_dim = A_DIM):
        # np.random.seed(random_seed)
        self.videos = fixed_vbr_env.VideoEnv(random_seed=random_seed)
        self.all_time, self.all_bw, self.all_names = self.load_trace(cooked_trace_folder)
        self.rand_idx = 0#np.random.randint(len(self.all_time))
        self.ptr = 0#np.random.randint(len(self.all_time[self.rand_idx]))
        self.bw_idx = self.all_bw[self.rand_idx]
        self.tm_idx = self.all_time[self.rand_idx]
        self.state = None
        self.act_idx = 0
        self.vbr_arr = []
        self.a_dim = a_dim
        self.idx = -1
    
    def reset(self):
        self.rand_idx = self.idx#np.random.randint(len(self.all_time))
        self.bw = self.all_bw[self.rand_idx]
        self.tm = self.all_time[self.rand_idx]
        self.ptr = 0#np.random.randint(np.maximum(len(self.tm) - 30, 1))
        sort_arr = np.sort(self.bw[self.ptr:])
        values, base = np.histogram(sort_arr, bins=S_LEN)
        values = np.array(values, dtype=np.float) / np.sum(values)
        base = base[1:]
        base = base / 6.0
        state = np.zeros([S_INFO, S_LEN])
        state[0] = np.array(values)
        state[1] = np.array(base)
        state[2, :] = 0.
        state[3, -1] = 0.
        self.act_idx = 0
        self.vbr_arr = []
        self.state = state
        self.video_feature = self.videos.get_features(self.idx)
        ret_state = {'network': state, 'video': self.video_feature}
        return ret_state

    def next(self):
        self.idx += 1
        self.video_feature = self.videos.get_features(self.idx)
        return self.idx < len(self.all_time)

    def get_trace_name(self):
        return self.all_names[self.idx]

    def step(self, act):
        state = self.state
        state[2, self.act_idx] = (act + 1.) / 20.
        self.act_idx += 1
        self.vbr_arr.append(act)
        #  find actual videos and ladders
        vmaf_arr, size_arr = self.videos.step(self.vbr_arr)
        bitrate_ladder = size_arr / 1024. / 1024. * 8. / 4.
        # bitrate_ladder = np.sort(bitrate_ladder)
        # bitrate_ladder *= 6.
        bw = np.array(self.bw[self.ptr:])
        tm = np.array(self.tm[self.ptr:])
        normalized_tm = tm - tm[0]
        inter_bw = np.interp(np.arange(normalized_tm[0], normalized_tm[-1], 4.0), normalized_tm, bw)
        util_arr = []
        counter = np.zeros(len(bitrate_ladder))
        for _bw in inter_bw:
            _bw = np.clip(_bw, 0., 6.)
            _select_idx = -1
            _max_ladder = None
            for _s_idx, _ladder in enumerate(bitrate_ladder):
                if _ladder <= _bw:
                    if _max_ladder is None or _ladder > _max_ladder:
                        _select_idx = _s_idx
                        _max_ladder = _ladder
            if _select_idx < 0:
                # nothing selected
                _s_ladder = bitrate_ladder[0]
                counter[0] += 1.
                _r = np.clip(1. - _s_ladder / (_bw + 1e-6), -100., 0.)
            else:
                _s_ladder = bitrate_ladder[_select_idx]
                counter[_select_idx] += 1.
                _r = np.clip(_s_ladder / (_bw + 1e-6), 0., 1.)
            util_arr.append(_r)
        self.state = state
        # to prob
        util_avg = np.sum(util_arr) / np.sum(counter) * 100.
        counter /= np.sum(counter)
        vmaf_avg = np.dot(counter, vmaf_arr)
        size_mean = 0. * 100. * np.mean(bitrate_ladder)
        rew  = vmaf_avg + util_avg - size_mean
        ret_state = {'network': state, 'video': self.video_feature}
        return ret_state, rew, self.act_idx == self.a_dim, {'vmaf': vmaf_avg, 'util': util_avg, 'size': 100. * np.mean(bitrate_ladder)}

if __name__ == "__main__":
    from itertools import combinations, permutations
    from tqdm import tqdm
    combo = list(combinations([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], 6))
    max_qoe = -100.0
    max_combo = None
    for pair in tqdm(combo):
        env = FixedNetworkEnvDiscrete()
        rews = []
        while env.next():
            obs = env.reset()
            ap_idx = 0
            for p in pair:
                obs, rew, done, info = env.step(p)
                if done:
                    rews.append(rew)
                    break
        _rew = np.mean(rews)
        if max_qoe < _rew:
            max_qoe = _rew
            max_combo = pair
            print(max_combo, max_qoe)
