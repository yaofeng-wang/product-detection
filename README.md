# Product-detection

Legend
- TL: Transfer learning
- FT: Fine tuning
- XNet: Xception model
- MNetV2: MobileNetV2 model
- DNet201: DenseNet201 model
- GAP: GlobalAveragePooling2D layer
- sm: softmax layer
- DA method 1: Data augmentation method mentioned in https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96
- RLRoP: ReduceLROnPlateau callback
- ES: Earlystopping callback
- MC: ModelCheckpoint callback
- LRS: Learning rate scheduler callback mentioned in https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu

Results

| Index | Model Architecture | Image size | Learning style  | Data Augmentation | Extra features | Callbacks, Label smoothing, Optimizer           | Public Score |
| ----- | ------------------ | ---------- | --------------- | ----------------- | -------------- | ----------------------------------------------- | ------------ |
| 01    | MNetV2 + GAP + sm  | 160x160x3  |  TL             |   -               | -              |  MC,label smoothing=0, adam                     | 0.62417      | 
| 02    | MNetV2 + GAP + sm  | 160x160x3  |  TL             |   -               | -              |  RLRoP+ES+MC,label smoothing=0, adam            | 0.66719      | 
| 03    | XNet + GAP + sm    | 512x512x3  |  FT             |   DA method 1     | -              |  LRS+ES+MC, label smoothing=0, adam             | 0.80258      |
| 04    | XNet + GAP + sm    | 512x512x3  |  FT             |   DA method 1     | -              |  LRS+ES+MC, label smoothing=0, adam, 3Fold CV   | 0.82026      |
| 05    | DNet201 + GAP + sm | 512x512x3  |  FT             |   DA method 1     | -              |  LRS+ES+MC, label smoothing=0, adam, 3Fold CV   | 0.82581      |
| 06    | DNet201 + GAP + sm | 512x512x3  |  FT             | Flips+GridMask    | -              |  RLRoP+ES+MC, label smoothing=0.1, adam         | 0.82607      |
| 07    | DNet201 + GAP + sm | 512x512x3  |  FT             | Flips+GridMask    | -              |  LRS+ES+MC, label smoothing=0.1, adam           | 0.81604      |
