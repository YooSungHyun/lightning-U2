import torch
import pytorch_lightning as pl
from utils.dataset_utils import get_concat_dataset, get_cache_file_path
from utils.config_loader import load_config
from argparse import Namespace
from models.u2.dataloader import AudioDataLoader
from typing import Union
from packaging import version
from utils.processor import AudioProcessor
from torchvision.transforms import Compose
from transformers.utils import is_datasets_available
import datasets
from models.u2.length_grouped_sampler import LengthGroupedSampler, DistributedLengthGroupedSampler
import math
import decimal


KERNEL_STRIDE_SIZE = {
    "linear": None,
    "conv2d": [[3, 3], [2, 2]],
    "conv2d6": [[3, 5], [2, 3]],
    "conv2d8": [[3, 3, 3], [2, 2, 2]],
}


class BiU2DataModule(pl.LightningDataModule):
    # https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html#datamodules
    def __init__(self, args: Namespace):
        super().__init__()
        self.pl_data_dirs = args.pl_data_dirs
        self.cache_main_dir = args.cache_main_dir
        self.num_shards = args.num_shards
        self.seed = args.seed
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.train_batch_drop_last = args.train_batch_drop_last
        self.per_device_eval_batch_size = args.per_device_eval_batch_size
        self.eval_batch_drop_last = args.eval_batch_drop_last
        self.accumulate_grad_batches = args.accumulate_grad_batches
        self.num_proc = args.num_proc
        self.group_by_length = args.group_by_length
        self.lengths = None
        self.input_name = args.input_name
        self.label_name = args.label_name
        self.length_column_name = args.length_column_name
        config = load_config(args.model_config)
        self.pad_token_id = config.data.audio.pad_token_id
        self.bos_token_id = config.data.text.bos_token_id
        self.log_mel_conf = config.data.audio.log_mel_conf
        self.normalize = config.data.audio.normalize
        self.speed_aug_conf = config.data.audio.speed_aug_conf
        self.spec_aug_conf = config.data.audio.spec_aug_conf
        self.spec_sub_conf = config.data.audio.spec_sub_conf
        self.spec_trim_conf = config.data.audio.spec_trim_conf
        self.filter_conformer_len_prob = config.data.audio.filter_conformer_conf
        if "input_layer" in config.model.encoder.keys():
            self.kernel_stride_size = KERNEL_STRIDE_SIZE[config.model.encoder.input_layer]
        else:
            self.kernel_stride_size = KERNEL_STRIDE_SIZE["conv2d"]
        self.audio_processor = AudioProcessor(
            self.speed_aug_conf,
            self.normalize,
            self.log_mel_conf,
            self.spec_aug_conf,
            self.spec_sub_conf,
            self.spec_trim_conf,
        )

    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            def torch_int_div(tensor1, tensor2):
                """
                A function that performs integer division across different versions of PyTorch.
                """
                parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)

                is_torch_less_than_1_8 = parsed_torch_version_base < version.parse("1.8.0")
                if is_torch_less_than_1_8:
                    return tensor1 // tensor2
                else:
                    return torch.div(tensor1, tensor2, rounding_mode="floor")

            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch_int_div(input_length - kernel_size, stride) + 1

        for kernel_size, stride in zip(self.kernel_stride_size[0], self.kernel_stride_size[1]):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def filter_conformer_ctc_len(
        self, batch, output_len_prob: float = 1.0, min_frame: float = 1.0, max_frame: float = 9999999999.0
    ):
        context = decimal.getcontext()
        context.rounding = decimal.ROUND_HALF_UP
        hop_length = int(self.log_mel_conf.sample_rate * self.log_mel_conf.window_stride_sec)

        speed_len_penalty = max(self.speed_aug_conf.speeds) if self.speed_aug_conf else 1.0
        # 배속 음성길이: 반올림(전체길이/배속)
        # 멜 스펙트로그램 변환 길이: 내림(전체길이/hop_length) + 1 -> 반올림은 반만 적용되는 리스크가 있어서, 내린다음 +1함
        speed_aug_mel_spected_len = (
            math.floor(int(round(decimal.Decimal(len(batch[self.input_name]) / speed_len_penalty), 0)) / hop_length)
            + 1
        )
        trim_len_penalty = self.spec_trim_conf.max_t if self.spec_trim_conf else 0
        # 트림 aug 음성길이: 전체길이 - trim max길이 -> trim은 1~max를 랜덤으로 샘플링하지만, 현재 시점에는 랜덤값을 알 수 없으므로, max 기준으로 필터링
        final_mel_spected_len = speed_aug_mel_spected_len - trim_len_penalty
        cnn_output_len = self._get_feat_extract_output_lengths(final_mel_spected_len)
        label_len = len(batch[self.label_name]["input_ids"])

        # 멜스펙 설정 기준 1프레임도 못만드는 녀석이면 날림
        wav_sec = len(batch[self.input_name]) / self.log_mel_conf.sample_rate
        ms_frame = wav_sec * 100  # 1 초가 100 frame
        frame_limit_flag = min_frame < ms_frame < max_frame

        # aug 다 적용해서 짧아진 길이 -> Conformer Convolution output으로 짧아진 길이 * 개발자 스레시홀드가 label 보다 커야만함.
        return frame_limit_flag and (cnn_output_len * output_len_prob).floor() > label_len

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage == "fit":
            self.train_datasets = get_concat_dataset(self.pl_data_dirs, "train")
            # before kspon: 612849
            if self.filter_conformer_len_prob:
                cache_file_name = get_cache_file_path(self.cache_main_dir, "syll_mel_len_filter", "train")
                self.train_datasets = self.train_datasets.filter(
                    self.filter_conformer_ctc_len,
                    num_proc=self.num_proc,
                    cache_file_name=cache_file_name,
                    fn_kwargs=self.filter_conformer_len_prob,
                )
            if self.group_by_length:
                if is_datasets_available() and isinstance(self.train_datasets, datasets.Dataset):
                    if self.length_column_name in self.train_datasets.column_names:
                        self.lengths = self.train_datasets[self.length_column_name]
                    else:
                        print("you must have 'lengths' column in your datasets. If you don't, it will VERY Slow")
                        self.lengths = [len(data[self.input_name]) for data in self.train_datasets]
                else:
                    self.lengths = None
            print(len(self.train_datasets))
            training_get_func = Compose(
                [
                    self.audio_processor.raw_audio_preprocess,
                    self.audio_processor.raw_to_logmelspect,
                    self.audio_processor.spectrogram_preprocess,
                    self.audio_processor.output_transpose,
                ]
            )
            self.train_datasets.set_transform(training_get_func)
            self.val_datasets = get_concat_dataset(self.pl_data_dirs, "dev")
            val_pre_processes = list()
            if self.normalize:
                val_pre_processes.append(self.audio_processor.mean_var_norm)
            val_pre_processes.append(self.audio_processor.raw_to_logmelspect)
            val_pre_processes.append(self.audio_processor.output_transpose)
            self.val_datasets.set_transform(Compose(val_pre_processes))

        if stage == "test":
            self.clean_datasets = get_concat_dataset(self.pl_data_dirs, "eval_clean")
            clean_pre_processes = list()
            if self.normalize:
                clean_pre_processes.append(self.audio_processor.mean_var_norm)
            clean_pre_processes.append(self.audio_processor.raw_to_logmelspect)
            clean_pre_processes.append(self.audio_processor.output_transpose)
            self.clean_datasets.set_transform(Compose(clean_pre_processes))
            self.other_datasets = get_concat_dataset(self.pl_data_dirs, "eval_other")
            other_pre_processes = list()
            if self.normalize:
                other_pre_processes.append(self.audio_processor.mean_var_norm)
            other_pre_processes.append(self.audio_processor.raw_to_logmelspect)
            other_pre_processes.append(self.audio_processor.output_transpose)
            self.other_datasets.set_transform(Compose(other_pre_processes))

    def train_dataloader(self):
        # setup에서 완성된 datasets를 여기서 사용하십시오. trainer의 fit() method가 사용합니다.
        generator = None
        if self.trainer.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.seed
            generator.manual_seed(seed)

        # Build the sampler.
        if self.group_by_length:
            model_input_name = self.input_name
            if self.trainer.world_size <= 1:
                train_sampler = LengthGroupedSampler(
                    self.per_device_train_batch_size * self.accumulate_grad_batches,
                    dataset=self.train_datasets,
                    lengths=self.lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                train_sampler = DistributedLengthGroupedSampler(
                    self.per_device_train_batch_size * self.accumulate_grad_batches,
                    dataset=self.train_datasets,
                    lengths=self.lengths,
                    model_input_name=model_input_name,
                    seed=self.seed,
                    drop_last=self.train_batch_drop_last,
                )
            return AudioDataLoader(
                dataset=self.train_datasets,
                batch_size=self.per_device_train_batch_size,
                num_workers=self.num_proc,
                pad_token_id=self.pad_token_id,
                label_name=self.label_name,
                pin_memory=True,
                sampler=train_sampler,
            )
        else:
            return AudioDataLoader(
                dataset=self.train_datasets,
                batch_size=self.per_device_train_batch_size,
                num_workers=self.num_proc,
                pad_token_id=self.pad_token_id,
                label_name=self.label_name,
                pin_memory=True,
            )

    def val_dataloader(self):
        # setup에서 완성된 datasets를 여기서 사용하십시오. trainer의 fit(), validate() method가 사용합니다.
        return AudioDataLoader(
            dataset=self.val_datasets,
            batch_size=self.per_device_eval_batch_size,
            num_workers=self.num_proc,
            pad_token_id=self.pad_token_id,
            label_name=self.label_name,
            pin_memory=True,
        )

    def test_dataloader(self):
        # setup에서 완성된 datasets를 여기서 사용하십시오. trainer의 test() method가 사용합니다.
        return [
            AudioDataLoader(
                dataset=self.clean_datasets,
                batch_size=1,
                num_workers=self.num_proc,
                pad_token_id=self.pad_token_id,
                label_name=self.label_name,
                pin_memory=True,
            ),
            AudioDataLoader(
                dataset=self.other_datasets,
                batch_size=1,
                num_workers=self.num_proc,
                pad_token_id=self.pad_token_id,
                label_name=self.label_name,
                pin_memory=True,
            ),
        ]

    def predict_dataloader(self):
        # setup에서 완성된 datasets를 여기서 사용하십시오. trainer의 predict() method가 사용합니다.
        pass
