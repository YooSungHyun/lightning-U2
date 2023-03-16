import torch
import random
from typing import Any, Dict
import torchaudio
from torchaudio.transforms import MelSpectrogram, TimeMasking, FrequencyMasking
import numpy as np

WINDOWS = {
    "hamming": torch.hamming_window,
    "hann": torch.hann_window,
    "blackman": torch.blackman_window,
    "bartlett": torch.bartlett_window,
}


class AudioProcessor:
    def __init__(
        self,
        speed_conf: Dict = None,
        normalize: bool = False,
        log_mel_conf: Dict = None,
        specaug_conf: Dict = None,
        specsub_conf: Dict = None,
        spectrim_conf: Dict = None,
    ):
        assert log_mel_conf, "transforming mel spec MUST NECESSARY"
        self.__speed_conf = speed_conf
        self.__normalize = normalize
        self.__log_mel_conf = log_mel_conf
        self.__specaug_conf = specaug_conf
        self.__specsub_conf = specsub_conf
        self.__spectrim_conf = spectrim_conf

    def __speed_perturb(self, batch, speeds=None):
        """Apply speed perturb to the data.
        Inplace operation.
        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            speeds(List[float]): optional speed
        Returns:
            Iterable[{key, wav, label, sample_rate}]
        """
        assert "input_values" in batch
        waveforms = [torch.FloatTensor(s).unsqueeze(0) for s in batch["input_values"]]
        # print([s.size() for s in waveforms])
        speed = random.choice(speeds)
        if speed != 1.0:
            results = list()
            for waveform in waveforms:
                wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                    waveform,
                    self.__log_mel_conf.sample_rate,
                    [["speed", str(speed)], ["rate", str(self.__log_mel_conf.sample_rate)]],
                )
                results.append(wav)
            batch["input_values"] = [s[0] for s in results]
        else:
            batch["input_values"] = [s[0] for s in waveforms]
        # print([s.size() for s in batch["input_values"]])
        return batch

    def mean_var_norm(self, batch):
        # print([s.size() for s in batch["input_values"]])
        datas = [torch.FloatTensor(s) for s in batch["input_values"]]
        # print([s.size() for s in datas])
        results = list()
        for data in datas:
            results.append((data - data.mean()) / torch.sqrt(data.var() + 1e-7))
        batch["input_values"] = results
        # print([s.size() for s in results])
        return batch

    def raw_to_logmelspect(self, batch):
        # window_size는 통상 사람이 변화를 느끼는 한계인 25ms을 기본으로 합니다 (0.025)
        # 16000 * 0.025 = 400
        win_length = int(np.ceil(self.__log_mel_conf.sample_rate * self.__log_mel_conf.window_size_sec))
        # n_fft는 학습에 쓰이기 위한 max_length로 보면 됩니다. 해당 길이만큼 pad가 붙고, 모델과 관련이 있다는 점에서
        # 2의 n승 혹은 512, 1024 등의 값을 주로 쓰는데, win_length보다 크거나 같으면 되므로 저는 같게 설정.
        n_fft = win_length
        # 얼마만큼 밀면서 자를것이냐, (얼마만큼 겹치게 할 것이냐) 1부터 숫자에서 win_length가 10 hop_length를 2로 하면, 1~10 -> 3~13 과 같은 형태로 밀림.
        hop_length = int(self.__log_mel_conf.sample_rate * self.__log_mel_conf.window_stride_sec)

        # print([s.size() for s in batch["input_values"]])
        # input have to (..., time)
        raw_audios = [torch.FloatTensor(s).unsqueeze(0) for s in batch["input_values"]]
        # print(raw_audio.size())
        results = list()
        for raw_audio in raw_audios:
            mel_spect = MelSpectrogram(
                sample_rate=self.__log_mel_conf.sample_rate,
                win_length=win_length,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=self.__log_mel_conf.n_mels,
                window_fn=WINDOWS.get(self.__log_mel_conf.window, WINDOWS["hamming"]),
            )(raw_audio)
            log_melspect = torch.log1p(mel_spect)
            results.append(log_melspect)
        # mel_spect shape (channel(mono(1), 2~3 etc), n_mels, time)
        # print([s.size() for s in results])
        batch["input_values"] = results
        return batch

    def __spec_augmentation(self, batch, freq_mask_para, time_mask_para, freq_mask_cnt, time_mask_cnt):
        # torch.random.manual_seed(self.seed)
        # datas shape: (channel, mel, seq)
        datas = batch["input_values"]
        # print([s.size() for s in data])
        # print(data, data.size())
        results = list()
        for data in datas:
            for _ in range(freq_mask_cnt):
                data = FrequencyMasking(freq_mask_param=freq_mask_para)(data)
            for _ in range(time_mask_cnt):
                data = TimeMasking(time_mask_param=time_mask_para)(data)
            results.append(data)
        # data shape: (channel, mel, seq)
        batch["input_values"] = results
        # print([s.size() for s in results])
        return batch

    def __spec_sub(self, batch, max_t=20, num_t_sub=3):
        """Do spec substitute
        Inplace operation
        ref: U2++, section 3.2.3 [https://arxiv.org/abs/2106.05642]
        Args:
            data: Iterable[{key, feat, label}]
            max_t: max width of time substitute
            num_t_sub: number of time substitute to apply
        Returns
            Iterable[{key, feat, label}]
        """
        assert "input_values" in batch
        datas = batch["input_values"]  # input (1, mel, time)
        # print([s.size() for s in datas])
        results = list()
        for data in datas:
            assert isinstance(data, torch.Tensor)
            y = data.clone().detach()
            max_frames = y.size(2)
            for i in range(num_t_sub):
                start = random.randint(0, max_frames - 1)
                length = random.randint(1, max_t)
                end = min(max_frames, start + length)
                # only substitute the earlier time chosen randomly for current time
                pos = random.randint(0, start)
                y[:, :, start:end] = data[:, :, start - pos : end - pos]
            results.append(y)
        batch["input_values"] = results  # y : tensor(1, mel, time)
        # print([s.size() for s in results])
        return batch

    def __spec_trim(self, batch, max_t=20):
        """Trim tailing frames. Inplace operation.
        ref: TrimTail [https://arxiv.org/abs/2211.00522]
        Args:
            data: Iterable[{key, feat, label}]
            max_t: max width of length trimming
        Returns
            Iterable[{key, feat, label}]
        """
        datas = batch["input_values"]  # input tensor(1, mel, time)
        results = list()
        for data in datas:
            # print(data.size())
            assert isinstance(data, torch.Tensor)
            max_frames = data.size(2)
            length = random.randint(1, max_t)
            # print(length)
            if length < max_frames / 2:
                y = data.clone().detach()[:, :, : max_frames - length]
                results.append(y)
            else:
                results.append(data)
            # print(y.size())
        batch["input_values"] = results  # y : tensor(1, mel, time)
        return batch

    def raw_audio_preprocess(self, raw: Dict[str, Any]):
        if self.__speed_conf:
            raw = self.__speed_perturb(batch=raw, **self.__speed_conf)
        # raw shape: tensor(raw_time)
        if self.__normalize:
            raw = self.mean_var_norm(batch=raw)
        # raw shape: tensor(raw_time)
        # print([s.size() for s in raw["input_values"]])
        return raw

    def spectrogram_preprocess(self, raw: Dict[str, Any]):
        if self.__specaug_conf:
            raw = self.__spec_augmentation(batch=raw, **self.__specaug_conf)
        if self.__specsub_conf:
            raw = self.__spec_sub(batch=raw, **self.__specsub_conf)
        if self.__spectrim_conf:
            raw = self.__spec_trim(batch=raw, **self.__spectrim_conf)
        # print(raw["input_values"], raw["input_values"].size())
        return raw

    def output_transpose(self, raw: Dict[str, Any]):
        raw["input_values"] = [s.squeeze(dim=0).transpose(1, 0) for s in raw["input_values"]]
        # print(raw["input_values"], raw["input_values"].size())
        return raw
