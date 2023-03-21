# lightning-U2

this code is followed by https://github.com/wenet-e2e/wenet and converted pytorch-lightning⚡

# Data & Datasets

## Must use pre-made HuggingFace Datasets!!

pre-made HuggingFace Style Datasets looks like this, <br />

> Dataset({ <br />
> &nbsp;&nbsp;&nbsp;&nbsp;features: ['input_values', 'labels', 'length'], <br />
> &nbsp;&nbsp;&nbsp;&nbsp;num_rows: 620000 <br />
> }) <br />

**input_values**: not normalized raw audio. librosa loaded 16000 sample_rate float32 <br />
**[labels][input_ids]**: ground truth label's tokenized integer value <br />
**length**: len(input_values). It is used HuggingFace style SmartBatching(group_by_length). <br />

**wenet used same like group_by_length**

- Mayby, If you used HuggingFace's Audio FeatureExtractor and Tokenizer, not feeling uncomfortable <br />
- `get_concat_dataset` is fit on sharded dataset <br />
  - **Example**
  - datasets root folder
    - train
      - 0
        - data
      - 1
        - data
    - dev
      - 0
        - data
    - eval_clean
      - 0
        - data

# datamodule

1. prepare_data()
   - Nothing to do
2. setup()
   - `get_concat_dataset`
     - get datasets on filelist
   - `filter_conformer_len_prob`
     - CTC loss have to longer input than label
       - I check on conformer cnn, trim aug etc...
   - `group_by_length` sampler
     - if you don't have length column, we calculate this block
   - `Compose`
     - It perfectly same on wenet
       - speed_perturb
       - mean_val_norm (like cmvn)
       - make log melspectrogram (but wenet used mfcc)
       - spec_aug, spec_sub, spec_trim
       - (mel,time) -> (time,mel) transpose
     - all process is doing each datasets called. so each epoch, you can meet new aug data
3. Each dataloader()
   - Only used `LengthGroupedSampler` in train loader
   - if `group_by_length==False` it just using torch lightning default

# model.py

All Process is same with wenet

but, Only different `optimizer_step`. because, i think this model training is so tense...

fp16, small epoch -> going gradient inf

fp16, big epoch -> train well

fp32, small epoch -> train well

and, label length is too much big, going gradient or loss inf

so, if gradient is something wrong, i don't update gradient