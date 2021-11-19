[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.6|3.7|3.8-blue.svg)](https://www.python.org)
[![Tensorflow](https://img.shields.io/badge/Tensorflow-2.2:2.5-red.svg)](https://www.tensorflow.org)
# Neural Audio Fingerprint for High-specific Audio Retrieval based on Contrastive Learning

<p align="left">
<img src="https://user-images.githubusercontent.com/26891722/124309350-8b6a0500-dba5-11eb-8059-d2005cbce260.gif" width="390">
&nbsp; &nbsp; &nbsp; &nbsp; <img src="https://user-images.githubusercontent.com/26891722/124309354-8e64f580-dba5-11eb-9271-df43c70a890a.png" width="350">
</p>

## About

* This is an official code and dataset release by authors (since July 2021) for reproducing [neural audio fingerprint](https://arxiv.org/abs/2010.11910).
* Previously, there was a [PyTorch implementation by Yi-Feng Chen](https://github.com/stdio2016/pfann). 
* :eight_spoked_asterisk: [Sound DEMO](https://mimbres.github.io/neural-audio-fp/) available now. 

## Requirements
Minimum:

- NVIDIA GPU with CUDA 10+
- 25 GB of free SSD space for mini dataset experiments

<details>
  <summary> More info </summary>

  #### System requirements to reproduce the ICASSP result

  - CPU with 8+ threads
  - NVIDIA GPU with 11+ GB V-memory
  - SSD free space 500+ GB for full-scale experiment 
  - `tar` extraction temporarily requires additional free space 440 GB. 

  #### Recommended batch-size for GPU
  | Device                                        |           Recommended BSZ |
  |-------------------------------------------------------------------|-------|
  | 1080ti, 2080ti (11GB), Titan X, Titan V (12GB), AWS/GCP V100(16 GB)| 320  |
  | Quadro RTX 6000 (24 GB), 3090 (24GB)          | 640                       |
  | V100v2 (32GB), AWS/GCP A100 (40 GB)                   | 1280                      |
  | ~~TPU~~                                  |~~5120~~                   |

  - The larger the BSZ, the higher the performance.
  - To allow the use of a larger BSZ than actual GPU memory, one trick is to
  remove `allow_gpu_memory_growth()` from the [run.py](run.py).

</details>


## Install

### [Docker](https://docs.docker.com/engine/install/ubuntu/)


```sh
docker pull mimbres/neural-audio-fp:latest
```

<details>
  <summary> Create a custom image from Dockerfile </summary>

  #### Requirements
  - NVIDIA driver >= 450.80.02
  - Docker > 20.0

  #### Create
  You can create an image through `Dockerfile` and `environment.yml`.

  ```sh
  git clone https://github.com/mimbres/neural-audio-fp.git
  cd neural-audio-fp
  docker build -t neural-audio-fp .
  ```
  
  #### Further information
  - Intel CPU users can remove `libopenblas` from Dockerfile.
  - `Faiss` and `Numpy` are optimized for Intel MKL.
  - Image size is about 12 GB or 6.43 GB (compressed).
  - To optimize GPU-based search speed, [install from the source](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md#building-from-source).

</details>


### [Conda](https://docs.anaconda.com/anaconda/install/index.html)
<details>
  <summary> Create a virtual environment via .yml </summary>

  #### Requirements

  - `NVIDIA driver >= 450.80.02`, `CUDA >= 11.0` and `cuDNN 8` [(Compatiability)](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html)
  - `NVIDIA driver >= 440.33`, `CUDA == 10.2` and `cuDNN 7` [(Compatiability)](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html)
  - [Anaconda3](https://docs.anaconda.com/anaconda/install/index.html) or Miniconda3 with Python >= 3.6

  #### Create
  After checking the requirements,

  ```sh
  git clone https://github.com/mimbres/neural-audio-fp.git
  cd neural-audio-fp
  conda env create -f environment.yml
  conda activate fp
  ```
    
</details>

<details>
  <summary> Create a virtual environment without .yml </summary>

  ```sh
  # Python 3.8: installing in the same virtual environment
  conda create -n YOUR_ENV_NAME 
  conda install -c anaconda -c pytorch tensorflow=2.4.1=gpu_py38h8a7d6ce_0 cudatoolkit faiss-gpu=1.6.5
  conda install pyyaml click matplotlib
  conda install -c conda-forge librosa
  pip install kapre wavio
  ```
  
</details>


<details>
  <summary> If your installation fails at this point and you don't want to build from source...:thinking: </summary>

  - Try installing `tensorflow` and `faiss-gpu=1.6.5` (not 1.7.1) in separate environments.
  
  ```sh
  #After creating a tensorflow environment for training...
  conda create -n YOUR_ENV_NAME
  conda install -c pytorch faiss-gpu=1.6.5
  conda install pyyaml, click
  ```
  
  Now you can run search & evaluation by
  
  ```
  python eval/eval_faiss.py --help
  
  ```
    
</details>

## Dataset

|     |Dataset-mini v1.1 (11.2 GB)  | Dataset-full v1.1 (443 GB) |
|:---:|:---:|:---:|
| tar |:eight_spoked_asterisk:[kaggle](https://www.kaggle.com/mimbres/neural-audio-fingerprint) / [gdrive](https://drive.google.com/file/d/1-eg7GhkOobhrTxFPMus7hVWNlR3AnPgE/view?usp=sharing) | [dataport(open-access)](http://ieee-dataport.org/open-access/neural-audio-fingerprint-dataset) |
| raw |[gdrive](https://drive.google.com/drive/folders/1JaEOX2b3M7J40N4mw2Sed9a4U3N5V-Zx?usp=sharing)|[gdrive](https://drive.google.com/drive/folders/1nOjfJ_WzNASGPa-dnRhnYklKaDegO6S-?usp=sharing)|

* The only difference between these two datasets is the size of 'test-dummy-db'.
  So you can first train and test with `Dataset-mini`. `Dataset-full` is for
  testing in 100x larger scale.
* You can download the `Dataset-mini` via `kaggle` [CLI](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication) (recommended).
  - Sign in [kaggle](https://kaggle.com) -> Account -> API -> Create New Token -> download `kaggle.json`

```
pip install --user kaggle
cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d mimbres/neural-audio-fingerprint

100%|███████████████████████████████████| 9.84G/9.84G [02:28<00:00, 88.6MB/s]
```

<details>

  <summary> Dataset installation </summary>

  This dataset includes all music sources, background noises, impulse-reponses
   (IR) samples that can be used for reproducing the ICASSP results.

  #### Directory location

  The default directory of the dataset is `../neural-audio-fp-dataset`. You can
   change the directory location by modifying `config/default.yaml`.

  ```
  .
  ├── neural-audio-fp-dataset
  └── neural-audio-fp
  ```

  #### Structure of dataset

  ```
  neural-audio-fp-dataset/
  ├── aug
  │   ├── bg         <=== Audioset, Pub/cafe etc. for background noise mix
  │   ├── ir         <=== IR data for microphone and room reverb simulatio
  │   └── speech     <=== subset of common-voice, NOT USED IN THE PAPER RESULT
  ├── extras
  │   └── fma_info   <=== Meta data for music sources.
  └── music
      ├── test-dummy-db-100k-full  <== 100K songs of full-lengths
      ├── test-query-db-500-30s    <== 500 songs (30s) and 2K synthesized queries
      ├── train-10k-30s            <== 10K songs (30s) for training
      └── val-query-db-500-30s     <== 500 songs (30s) for validation/mini-search
  ```

  The data format is `16-bit 8000 Hz PCM Mono WAV`. `README.md` and `LICENSE` is
     included in the dataset for more details.

</details>


<details>
  
  <summary> Checksum for Dataset-full </summary>
  
  Install `checksumdir`.
  
  ```
  pip install checksumdir
  ```
  
  Compare checksum.
  
  ```
  checksumdir -a md5 neural-audio-fp-dataset
  # aa90a8fbd3e6f938cac220d8aefdb134
  
  checksumdir -a sha1 neural-audio-fp-dataset
  # 5bbeec7f5873d8e5619d6b0de87c90e180363863d
  ```
  
</details>


## Quickstart

There are 3 basic `COMMAND` s for each step.

```python
# Train
python run.py train CHECKPOINT_NAME

# Generate fingreprint
python run.py generate CHECKPOINT_NAME

# Search & Evalutaion (after generating fingerprint)
python run.py evaluate CHECKPOINT_NAME CHECKPOINT_INDEX
```

Help for `run.py` client and its commands.

```python
python run.py --help
python run.py COMMAND --help
```

## More Features

Click to expand each topic.

<details>
  <summary> Managing Checkpoint </summary>

  ```python
  python run.py train CHECKPOINT_NAME CHECKPOINT_INDEX
  ```

  - If `CHECKPOINT_INDEX` is not specified, the training will resume from the
   latest checkpoint.
  - In `default` configuration, all checkpoints are stored in
  `logs/checkpoint/CHECKPOINT_NAME/ckpt-CHECKPOINT_INDEX.index`.

</details>

<details>
  <summary> Training </summary>

  ```python
  python run.py train CHECKPOINT --max_epoch=100 -c default
  ```

  Notes:

  - Check batch-size that fits on your device first.
  - The `default` config is set `TR_BATCH_SZ`=120 with `OPTIMIZER`=`Adam`.
  - For `TR_BATCH_SZ` >= 240, `OPTIMIZER`=`LAMB` is recommended.
  - For `TR_BATCH_SZ` >= 1280, `LR`=`1e-4` can be too small.
  - In NTxent loss function, the best temperature parameter `TAU` is in the
    range of [0.05, 0.1].
  - Augmentation strategy is quite important. This topic deserves further discussion.

</details>


<details>
  <summary> Config File </summary>

  The config file is located in `config/CONFIG_NAME.yaml`.
  You can edit `directory location`, `data selection`, hyperparameters for
   `model` and `optimizer`, `batch-size`, strategies for time-domain and
   spectral-domain `augmentation chain`, etc. After training, it is important
    to keep the config file in order to restore the model.

  ```python
  python run.py COMMAND -c CONFIG
  ```

  When using `generate` command, it is important to use the same config that was used
  in training.

</details>

<details>
  <summary> Fingerprint Generation </summary>

  ```python
  python run.py generate CHECKPOINT_NAME # from the latest checkpoint
  python run.py generate CHECKPOINT_NAME CHECKPOINT_INDEX -c CONFIG_NAME
  ```
  ```sh
  # Location of the generated fingerprint
  .
  └──logs
     └── emb
         └── CHECKPOINT_NAME
             └── CHECKPOINT_INDEX
                 ├── db.mm
                 ├── db_shape.npy
                 ├── dummy_db.mm
                 ├── dummy_db_shape.npy
                 ├── query.mm
                 └── query_shape.npy
  ```
  By `default` config, `generate` will generate embeddings (or fingerprints)
   from 'dummy_db', `test_query` and `test_db`. The generated embeddings will 
   be located in `logs/emb/CHECKPOINT_NAME/CHECKPOINT_INDEX/**.mm` and 
   `**.npy`.

  - `dummy_db` is generated from the 100K full-length dataset.
  - In the `DATASEL` section of config, you can select options for a pair of
   `db`  and `query` generation. The default is `unseen_icassp`, which uses a
    pre-defined test set.
  - It is possilbe to generate only the `db` and `query` pairs by 
  `--skip_dummy` option. This is a frequently used option to avoid overwriting 
    the most time-consuming `dummy_db` fingerprints in every experiment.   
  - It is also possilbe to generate embeddings (or fingreprints) from your
   custom source.

  ```python
  python run.py generate --source SOURCE_ROOT_DIR --output FP_OUTPUT_DIR --skip_dummy # for custom audio source
  python run.py generate --help # more details...
  ```

</details>

<details>
  <summary> Search & Evaluation </summary>

  The following command will construct a `faiss.index` from the generated
  embeddings or fingerprints located at
  `logs/emb/CHECKPOINT_NAME/CHECKPOINT_INDEX/`.

  ```python
  # faiss-gpu
  python run.py evaluate CHECKPOINT_NAME CHECKPOINT_INDEX [OPTIONS]

  # faiss-cpu
  python run.py evaluate CHECKPOINT_NAME CHECKPOINT_INDEX --nogpu

  ```

  In addition, you can choose one of the `--index_type` (default is `IVFPQ`) 
  from the table below:

  | Type of index | Description |
  | --- | --- |
  | `l2` | *L2* distance|
  | `ivf` | Inverted File Index (IVF) |
  | `ivfpq` | Product Quantization (PQ) with IVF [:book:](https://arxiv.org/pdf/1702.08734) |
  | `ivfpq-rr` | IVF-PQ with re-ranking |
  | ~~`ivfpq-rr-ondisk`~~ | ~~IVF-PQ with re-ranking on disk search~~ |
  | `hnsw` | Hierarchical Navigable Small World [:book:](https://arxiv.org/abs/1603.09320) |

  ```python
  python run.py evaluate CHECKPOINT_NAME CHECKPOINT_INDEX --index_type IVFPQ
  ```

  Currently, few options for `Faiss` settings are available in `run.py` client.
  Instead, you can directly run:

  ```
  python eval/eval_faiss.py EMB_DIR --index_type IVFPQ --kprobe 20 --nogpu
  python eval/eval_faiss.py --help

  ```

  Note that `eval_faiss.py` does not require `Tensorflow`.

</details>

<details>
  <summary> Tensorboard </summary>

  Tensorboard is enabled by default in the `['TRAIN']` section of the config file.

  ```python
  # Run Tensorboard
  tensorboard --logdir=logs/fit --port=8900 --host=0.0.0.0
  ```

</details>


## Build DB & Search

Here is an overview of the system for building and retrieving the database.
The system and 'matcher' algorithm are not detailed in the paper.
But it's very simple as in this [code](eval/eval_faiss.py#L214).

<p align="left">
<img src="https://user-images.githubusercontent.com/26891722/124311508-d5a0b580-dba8-11eb-9d00-ba298fc1ea54.png" width="700">
</p>


## Plan

* Now working on `tf.data`-based new data pipeline for multi-GPU and TPU support.
* One page Colab demo.
* This project is currently based on [Faiss](https://github.com/facebookresearch/faiss), which provides the fastest large-scale vector searches.
* [Milvus](https://github.com/milvus-io/milvus) is also worth watching as it is an active project aimed at industrial scale vector search.


## Augmentation Demo and Scoreboard

[Augmentation demo](https://mimbres.github.io/neural-audio-fp/) was generated by [dataset2wav.py](extras/dataset2wav.py).

## External links

* (Unofficial) [PyTorch implementation by Yi-Feng Chan](https://github.com/stdio2016/pfann).

## Acknowledgement

This project has been supported by the TPU Research Cloud (TRC) program.

## Cite

```markdown
@conference {chang2021neural,
    author={Chang, Sungkyun and Lee, Donmoon and Park, Jeongsoo and Lim, Hyungui and Lee, Kyogu and Ko, Karam and Han, Yoonchang},
    title={Neural Audio Fingerprint for High-specific Audio Retrieval based on Contrastive Learning},
    booktitle={International Conference on Acoustics, Speech and Signal Processing (ICASSP 2021)},
    year = {2021}
}
```
