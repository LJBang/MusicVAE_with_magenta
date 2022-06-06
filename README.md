# MusicVAE_with_magenta
Magenta를 이용해 4마디 드럼 연주를 생성하는 예제입니다.

[Notebook 실습](./4bar_drums.ipynb)

## Requrements
```sh
tensorflow
magenta

###### requirements for midi play ######
pyfluidsynth
```  
magenta 라이브러리의 설치의 자세한 사항은 [여기](https://github.com/magenta/magenta#installation)를 참고하세요.  

## Dataset
[Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove)을 사용하였습니다.  

### MIDI Files
MIDI 데이터의 경우 학습을 위해 `.tfrecord`파일로 변환해주어야 하는데,  
`make_tfrecord.py`파일을 이용해 해당 함수를 불러와서 변환하였습니다.  
**Magenta**의 해당 모듈의 자세한 사용법은 [여기](https://github.com/magenta/magenta/blob/main/magenta/scripts/README.md)를 참고하세요.  

```sh
# command
python3 make_tfrecord.py \
 --input_dir=dataset/groove/ \
 --output_file=dataset/drummer.tfrecord \
 --recursive
```

### TF Dataset
해당 데이터는 TFDS에서도 사용할 수 있으므로 아래와 같이 불러올 수 있습니다.

```py
# python
import tensorflow_datasets as tfds

dataset = tfds.load(
    name="groove/full-midionly",
    split=tfds.Split.TRAIN,
    try_gcs=True)
```  

## 모델
`train.py`를 따로 만들어 `MusicVAE`모델을 정의하고, 학습할 수 있도록 만들었습니다.  
4마디 드럼 학습을 위한 모델은 `flat`과 `hierdec_small`, `hierdec_large` 세 가지로 구현하였으며 그 차이는 계층형 디코더의 유무와 lstm layer의 크기입니다.  

large 모델의 경우 아래와 같은 사양을 가집니다.  
```
Encoder: 2 Layer of 1024 size BidirectionalLSTM
Decoder: 2 Layer of 512 size CategoricalLSTM
latent vector: 256 size

Converter: DrumConverter
```  

small 모델의 경우 아래와 같은 사양을 가집니다.  
```
Encoder: 2 Layer of 512 size BidirectionalLSTM
Decoder: 2 Layer of 256 size CategoricalLSTM
latent vector: 256 size

Converter: DrumConverter
```  

## 학습
`train.py`를 커맨드로 실행해 학습합니다.  

```sh
# command
python3 train.py \
 --config=hierdec-drums_4bar \
 --run_dir=result/hier/ \
 --num_steps=50 \
 --mode=train \
 --examples_path=dataset/notesequences.tfrecord
```  

파라미터에 대한 자세한 사항은 [여기](https://github.com/magenta/magenta/tree/main/magenta/models/music_vae#training-your-own-musicvae)를 참고하세요.  

만약 tfrecord파일이 아니라 tfds에서 데이터를 불러와서 사용할 경우 아래와 같습니다.  

```sh
# command
python3 train.py \
 --config=hierdec-drums_4bar \
 --run_dir=result/hier/ \
 --num_steps=50 \
 --mode=train \
 --tfds_name=groove/full-midionly \
```  

## samples
[samples](./samples)  
`small_model_50epoch_drum_sample`: 계층형 디코더를 사용한 small모델로 50번의 학습을 통해 만들어진 MIDI 샘플입니다.  
`large_model_1000epoch_drum_sample`: 계층형 디코더를 사용한 large모델로 1000번의 학습을 통해 만들어진 MIDI 샘플입니다.  

