# Data folder

This project expects the standard `ImageFolder` layout:

```
data/
  train/
    0/  <png/jpg images of digit 0>
    1/  <images of digit 1>
    ...
    9/  <images of digit 9>
  val/
    0/
    1/
    ...
    9/
```

Each subfolder name is the class label (the digit). Images can be grayscale or RGB;
they are automatically resized and normalized during training.

## Don't have data yet?

Leave `data.auto_download: true` in `configs/default.yaml`. The first time you train,
the MNIST dataset is downloaded and written into the `train/` and `val/` folders above
automatically.
