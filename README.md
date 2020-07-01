# product-detection

Legend
- TL: Transfer learning
- FT: Fine tuning
- XNet: Xception model
- MNetV2: MobileNetV2 model
- DNet201: DenseNet201 model
- GAP: GlobalAveragePooling2D layer
- sm: softmax layer
- DA method 1: Data augmentation method mentioned [here](https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96)
- RLRoP: ReduceLROnPlateau callback
- ES: Earlystopping callback
- MC: ModelCheckpoint callback
- LRS: Learning rate scheduler callback mentioned [here](https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu)

Results

| Index | Model Architecture | Image size | TL/FT  | Data Augmentation | Extra features | Callbacks, Label smoothing, Optimizer           | Public Score |
| ----- | ------------------ | ---------- | -------| ----------------- | -------------- | ----------------------------------------------- | ------------ |
| 01    | MNetV2 + GAP + sm  | 160x160x3  |  TL    |   -               | -              |  MC,label smoothing=0, adam                     | 0.62417      | 
| 02    | MNetV2 + GAP + sm  | 160x160x3  |  TL    |   -               | -              |  RLRoP+ES+MC,label smoothing=0, adam            | 0.66719      | 
| 03    | XNet + GAP + sm    | 512x512x3  |  FT    |   -               | -              |  LRS+ES+MC, label smoothing=0, adam             |    ?         |
| 03    | XNet + GAP + sm    | 512x512x3  |  FT    |   DA method 1     | -              |  LRS+ES+MC, label smoothing=0, adam             | 0.80258      |
| 04    | XNet + GAP + sm    | 512x512x3  |  FT    |   DA method 1     | -              |  LRS+ES+MC, label smoothing=0, adam, 3Fold CV   | 0.82026      |
| 05    | DNet201 + GAP + sm | 512x512x3  |  FT    |   DA method 1     | -              |  LRS+ES+MC, label smoothing=0, adam, 3Fold CV   | 0.82581      |
| 06    | DNet201 + GAP + sm | 512x512x3  |  FT    | Flips+GridMask    | -              |  RLRoP+ES+MC, label smoothing=0.1, adam         | 0.82607      |
| 07    | DNet201 + GAP + sm | 512x512x3  |  FT    | Flips+GridMask    | -              |  LRS+ES+MC, label smoothing=0.1, adam           | 0.81604      |
| 08    | DNet201 + GAP + sm | 512x512x3  |  FT    | Flips+GridMask    | -              |  LRS+ES+MC, label smoothing=0.2, adam           | 0.80390     |

Things that helped:
1. TPU >> GPU >> CPU.
2. More complex model such as DNet/XNet compared to MNetV2. Training acc was able to increase from ~0.7 to ~0.99. Although, difficult to say DNet or XNet is better.
3. FT >> TL.
4. Data agumentation. Although, difficult to say kind of DA is better.
5. KFold CV helps slightly

Things that does not seem to help:
1. Label smoothing

Things to note:
1. validation set seems to resemble closely to test set. validation acc is usually +0.01 to +0.02 of public score.


Reference Kaggle notebooks:
1. [Getting started with 100+ flowers on TPU](https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu): Basic template for using TPU on classification problem with data augmentation: random flip left right.
2. [Rotation Augmentation GPU/TPU - [0.96+]](https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96): Using TPU on classification problem with data augmentation: random rotation, shearing, zoom, and shift, flip left right.

Reference online articles:
1. [TFRecord and tf.Example](https://www.tensorflow.org/tutorials/load_data/tfrecord): Tensorflow's official tutorial on TFRecords.
2. [Tensorflow Records? What they are and how to use them](https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564): In-depth explanation on writing/reading TFRecords.

Reference papers:
1. [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
