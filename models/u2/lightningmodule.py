import os
import torch
import pytorch_lightning as pl
from utils.cmvn import load_cmvn
from utils.config_loader import load_config
from models.u2.transformer.cmvn import GlobalCMVN
from models.u2.transformer.encoder import ConformerEncoder
from models.u2.transformer.decoder import BiTransformerDecoder
from models.u2.transformer.ctc import CTC
from models.u2.transformer.asr_model import ASRModel
from torchmetrics import WordErrorRate, CharErrorRate
from transformers import Wav2Vec2CTCTokenizer


class BiU2(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        config_cls = load_config(args.model_config)
        if os.path.isfile(os.path.join(args.output_dir, "tokenizer_config.json")):
            self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(os.path.join(args.output_dir))
        else:
            self.tokenizer = Wav2Vec2CTCTokenizer(args.vocab_path)
            os.makedirs(args.output_dir, exist_ok=True)
            self.tokenizer.save_pretrained(args.output_dir)
        if "cmvn" in config_cls.keys():
            mean, istd = load_cmvn(config_cls.cmvn.cmvn_file, config_cls.cmvn.is_json_cmvn)
            global_cmvn = GlobalCMVN(torch.from_numpy(mean).float(), torch.from_numpy(istd).float())
        else:
            global_cmvn = None
        encoder = ConformerEncoder(global_cmvn=global_cmvn, **config_cls["model"]["encoder"])
        assert 0.0 < config_cls.model.reverse_weight < 1.0
        assert config_cls.model.decoder.r_num_blocks > 0
        decoder = BiTransformerDecoder(
            vocab_size=len(self.tokenizer), encoder_output_size=encoder.output_size(), **config_cls["model"]["decoder"]
        )
        ctc = CTC(
            len(self.tokenizer),
            encoder.output_size(),
            reduction=config_cls.model.ctc_reduction,
            zero_infinity=config_cls.model.ctc_zero_inf,
        )
        self.calc_wer = WordErrorRate()
        self.calc_cer = CharErrorRate()
        self.model = ASRModel(
            vocab_size=len(self.tokenizer),
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            ctc_weight=config_cls.model.ctc_weight,
            ignore_id=config_cls.data.text.pad_token_id,
            bos_token_id=config_cls.data.text.bos_token_id,
            eos_token_id=config_cls.data.text.eos_token_id,
            reverse_weight=config_cls.model.reverse_weight,
            lsm_weight=config_cls.model.lsm_weight,
            length_normalized_loss=config_cls.model.length_normalized_loss,
        )

    def training_step(self, batch, batch_idx):
        input_audios, audio_lengths, targets, target_lengths = batch
        losses = self.model(input_audios, audio_lengths, targets, target_lengths)
        self.log("train_loss", losses["loss"], sync_dist=True)
        self.log("train_att_loss", losses["loss_att"], sync_dist=True)
        self.log("train_ctc_loss", losses["loss_ctc"], sync_dist=True)
        return {"loss": losses["loss"]}

    def validation_step(self, batch, batch_idx):
        input_audios, audio_lengths, targets, target_lengths = batch
        losses = self.model(input_audios, audio_lengths, targets, target_lengths)
        preds_tokens, best_log_score = self.model.recognize(input_audios, audio_lengths, 1)
        return {"loss": losses["loss"], "preds_tokens": preds_tokens, "labels": targets}

    def validation_epoch_end(self, validation_step_outputs):
        loss_mean = torch.tensor([x["loss"] for x in validation_step_outputs], device=self.device).mean()
        preds = list()
        labels = list()
        for x in validation_step_outputs:
            preds.extend(x["preds_tokens"])
            labels.extend(x["labels"])
        preds = self.tokenizer.batch_decode(preds)
        labels = self.tokenizer.batch_decode(labels)
        wer = self.calc_wer(preds, labels)
        cer = self.calc_cer(preds, labels)
        # sync_dist use follow this url
        # if using torchmetrics -> https://torchmetrics.readthedocs.io/en/stable/
        # if not using torchmetrics -> https://github.com/Lightning-AI/lightning/discussions/6501
        self.log("val_loss", loss_mean, sync_dist=True)
        self.log("val_wer", wer, sync_dist=True)
        self.log("val_cer", cer, sync_dist=True)
        # self.log_dict(metrics, sync_dist=(self.device != "cpu"))

    def predict_step(self, batch, batch_idx):
        features, labels, feature_lengths, label_lengths = batch
        logits = self(features)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [{"params": [p for p in self.parameters()], "name": "OneCycleLR"}],
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.args.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.args.warmup_ratio,
            epochs=self.trainer.max_epochs,
            final_div_factor=self.args.final_div_factor
            # steps_per_epoch=self.steps_per_epoch,
        )
        lr_scheduler = {"interval": "step", "scheduler": scheduler, "name": "AdamW"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
