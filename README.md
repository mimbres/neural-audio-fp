[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.6|3.7|3.8-blue.svg)](https://www.python.org)
[![Tensorflow](https://img.shields.io/badge/Tensorflow-2.2:2.5-red.svg)](https://www.tensorflow.org)
# Neural Audio Fingerprint for High-specific Audio Retrieval based on Contrastive Learning

<p align="left">
<img src="https://user-images.githubusercontent.com/26891722/124309350-8b6a0500-dba5-11eb-8059-d2005cbce260.gif" width="390">
&nbsp; &nbsp; &nbsp; &nbsp; <img src="https://user-images.githubusercontent.com/26891722/124309354-8e64f580-dba5-11eb-9271-df43c70a890a.png" width="350">
</p>

## Intro

This is the code and dataset release (on July 2021) of the [neural audio fingerprint](https://arxiv.org/abs/2010.11910).


## Requirements
Minimum:

- NVIDIA GPU with CUDA 10+
- Free disk space 20 GB for experiments with **mini** dataset

<details>
  <summary> More info </summary>

  #### System requirements to reproduce ICASSP result

  - CPU with 8+ threads
  - NVIDIA GPU with 11+ GB V-memory
  - SSD free space 500+ GB for full-scale experiment

  #### Recommended batch-size for GPU
  | Device                                        |           Recommended BSZ |
  |-------------------------------------------------------------------|-------|
  | 1080ti, 2080ti (11GB), Titan X, Titan V (12GB), AWS/GCP V100(16 GB)| 320  |
  | Quadro RTX 6000 (24 GB), 3090 (24GB)          | 640                       |
  | V100v2 (32GB), AWS/GCP A100 (40 GB)                   | 1280                      |
  | ~~TPU v3-8~~                                  |~~5120~~                   |

  - The larger the BSZ, the higher the performance.
  - For BSZ < 240, Adam optimizer is recommended.
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

  You can create image through `Dockerfile` and `environment.yml`.

  ```sh
  git clone https://github.com/mimbres/neural-audio-fp.git
  cd neural-audio-fp
  docker build -t neural-audio-fp .
  ```

  #### Requirements
  - NVIDIA driver >= 450.80.02
  - Docker > 20.0

  #### Further information
  - Intel CPU users can remove `libopenblas` from Dockerfile.
  - `Faiss` and `Numpy` are optimized for Intel MKL.
  - Image size is about 12 GB or compressed 6.43 GB.
  - To optimize GPU-based search speed, [install from the source](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md#building-from-source).
  - **RTX 3090** and **Cloud A100** users are highly recommended to build `Faiss-gpu` from source.

</details>


### [Conda](https://docs.anaconda.com/anaconda/install/index.html)
<details>
  <summary> Create a virtual environment via Conda </summary>

  #### Requirements

  - `NVIDIA driver >= 450.80.02`, `CUDA >= 11.0` and `cuDNN 8` [(Compatiability)](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html)
  - `NVIDIA driver >= 440.33`, `CUDA == 10.2` and `cuDNN 7` [(Compatiability)](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html)
  - [Anaconda3](https://docs.anaconda.com/anaconda/install/index.html) or Miniconda3 with Python >= 3.6

  After checking the requirements,

  ```sh
  git clone https://github.com/mimbres/neural-audio-fp.git
  cd neural-audio-fp
  conda env create -f environment.yml
  conda activate fp
  ```
</details>

  If your installation so far fails and you don't want to build from source:

  - Try `tensorflow` and `faiss-gpu=1.6.5` (not 1.7.1) in separate environments.

  ```sh
  # python 3.8
  conda install -c anaconda -c pytorch tensorflow=2.4.1=gpu_py38h8a7d6ce_0 cudatoolkit faiss-gpu=1.6.5
  conda install pyyaml click matplotlib
  conda install -c conda-forge librosa
  pip install kapre wavio
  ```


## Dataset

Dataset-mini (10.7GB): [Dataport](https://ieee-dataport.org/documents/neural-audio-fingerprint-dataset-mini) [Gdrive](https://drive.google.com/file/d/1bRXfrGQk3KNAEHdjGeOBHJTKJIAIUtU1/view?usp=sharing)
Dataset-full (414GB) : [Dataport]() *now uploading...*

* The only difference between these two datasets is the dummy 'set.Dataset-full'
 has full-scale dummy songs. 

<details>

  <summary> Dataset overview </summary>

  This dataset includes all music sources, background noise, impulse-reponses
   (IR) samples that can be used for reproducing the ICASSP results.

  #### Directory location

  The default directory of the dataset is '../fingerprint_dataset'. You can change the directories in config/default.yaml.

  ```
  .
  ├── fingerprint_dataset
  └── neural-audio-fp
  ```

  #### Structure of dataset

  ```
  fingerprint_dataset_icassp2021/
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

  The data format is `16-bit 8000 Hz PCM Mono WAV`. README.md and LICENSE is
     included in the dataset for more details.

</details>


## Quickstart

There are 3 basic `COMMAND`s.

```python
# Train
python run.py train CHECKPOINT_NAME

# Generate fingreprint
python run.py generate CHECKPOINT_NAME

# Search & Evalutaion (after generating fingerprint)
python run.py evaluate CHECKPOINT_NAME CHECKPOINT_INDEX
```

Help menu for `run.py` client and its commands.

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
  - In `default` configuration, every checkpoints are stored in
  `logs/checkpoint/CHECKPOINT_NAME/ckpt-CHECKPOINT_INDEX.index`.

</details>

<details>
  <summary> Training </summary>

  ```python
  python run.py train CHECKPOINT --max_epoch=100 -c default
  ```

  Notes:

  - Check batch-size that fits on your device first.
  - The `default` config is set `TR_BATCH_SZ`=120 with `OTIMIZER`=`Adam`.
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

  In `generate` command, it is important to use the same config that was used
  in training.

</details>

<details>
  <summary> Fingerprint Generatation </summary>

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
  By `default` config, `generate` will generate embeddings (or fingerprints) from
   'dummy_db', `test_query' and 'test_db'. The generated embeddings will be
   located in `logs/emb/CHECKPOINT_NAME/CHECKPOINT_INDEX/**.mm` and `**.npy`.

  - `dummy_db` is generated from the 100K full-length dataset.
  - In the `DATASEL` section of config, you can set option for `test_db` and
   `test_query` generation. The default is `unseen_icassp`, which uses a
    pre-defined test set.

  It is possilbe to generate embeddings (or fingreprints) from your custom source.

  ```python
  python run.py generate --source SOURCE_ROOT_DIR --output FP_OUTPUT_DIR # for custom audio source
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

  In addition, you can choose one of the `--index_type` (default is `IVFPQ`) from the table below.

  | Type of index | Description |
  | --- | --- |
  | `l2` | *L2* distance|
  | `ivf` | Inverted File Index (IVF) |
  | `ivfpq` | Product Quantizaion (PQ) with IVF [:book:](https://arxiv.org/pdf/1702.08734) |
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

Here is an overview of the system for building and retrieving database.
The system and 'matcher' algorithm are not detailed in the paper.
But it's very simple as in this [code]().

<p align="left">
<img src="https://user-images.githubusercontent.com/26891722/124311508-d5a0b580-dba8-11eb-9d00-ba298fc1ea54.png" width="700">
</p>


## Plan (?)

* Now working on `tf.data`-based new data pipeline for multi-GPU and TPU support.
* One page demo using Colab.
* This project is currently based on [Faiss](https://github.com/facebookresearch/faiss), which provides the fastest large-scale vector searches.
* [Milvus](https://github.com/milvus-io/milvus) is also worth watching as it is an active project aimed at industrial scale vector search.
* Someone can PR :)


## Augmentation Demo and Scoreboard

Synthesized audio samples from `dataset_augmentor_demo.py` TBA.

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