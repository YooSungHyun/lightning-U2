from utils.processor import AudioProcessor
from utils.dataset_utils import get_concat_dataset
from utils.config_loader import load_config
from torchvision.transforms import Compose
import numpy as np
import math
from tqdm import tqdm

config = load_config("./config/conformer_u2++.yaml")
log_mel_conf = config.data.audio.log_mel_conf
normalize = config.data.audio.normalize
speed_aug_conf = None
spec_aug_conf = config.data.audio.spec_aug_conf
filter_conformer_len_prob = config.data.audio.filter_conformer_conf
spec_sub_conf = config.data.audio.spec_sub_conf
spec_trim_conf = None
win_length = int(np.ceil(log_mel_conf.sample_rate * log_mel_conf.window_size_sec))
n_fft = win_length
hop_length = int(log_mel_conf.sample_rate * log_mel_conf.window_stride_sec)

audio_processor = AudioProcessor(
    speed_aug_conf,
    normalize,
    log_mel_conf,
    spec_aug_conf,
    spec_sub_conf,
    spec_trim_conf,
)
ori_datasets = get_concat_dataset(
    ["/ext_disk/stt/datasets/fine-tuning/42maru/data-KsponSpeech-42maru-not-normal-20"], "dev"
)
train_datasets = get_concat_dataset(
    ["/ext_disk/stt/datasets/fine-tuning/42maru/data-KsponSpeech-42maru-not-normal-20"], "train"
)
training_get_func = Compose(
    [
        audio_processor.raw_audio_preprocess,
        audio_processor.raw_to_logmelspect,
        audio_processor.spectrogram_preprocess,
        audio_processor.output_transpose,
    ]
)
train_datasets.set_transform(training_get_func)
val_datasets = get_concat_dataset(
    ["/ext_disk/stt/datasets/fine-tuning/42maru/data-KsponSpeech-42maru-not-normal-20"], "dev"
)
val_datasets.set_transform(
    Compose([audio_processor.mean_var_norm, audio_processor.raw_to_logmelspect, audio_processor.output_transpose])
)


result = list()
for data in tqdm(train_datasets):
    flag = data["input_values"].size(0) == (math.floor(data["length"] / hop_length) + 1)
    if not flag:
        print("다른애 발생")
        break


# def test(batch):
#     flag = batch["input_values"].size(0) == math.floor(batch["length"] / hop_length) + 1
#     batch["flag"] = flag
#     return batch


# train_datasets = train_datasets.map(test, num_proc=12)
# print(train_datasets)
