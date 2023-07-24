# Kaggle-Plant-Seedlings-Classification

## Dataset

Donwload dataset [here](https://www.kaggle.com/competitions/plant-seedlings-classification/data). Unzipping the files to the current directory.

## Traditional Methods

To use SIFT, run the following command:

```
python SIFT_BOW_SVM.py
```

To use HOG, run the following command:

```
python HOG_SVM.py
```

To use HOG + SIFT, run the following command:

```
python HOG_SIFT_BOW_SVM.py
```

## Deep Learning Methods

python dl.py   --exp_name EXP_NAME   [--batchsize BATCHSIZE] [--num_epochs NUM_EPOCHS] [--lr LR] [--model MODEL] [--optimizer OPTIMIZER] [--weight_decay WEIGHT_DECAY]

Arguments:

--exp_name (-e) str: This is a required parameter that determines the output filename, and the file will be generated in the 'results' directory.

--batchsize (-b) int: Default is 64.

--num_epochs (-n) int: Default is 20.

--lr (-l) float: Default is 1e-4.

--model (-m) str: Default is "ResNet," could be "ResNet," "VGG," "DenseNet," or "MLP."

--optimizer (-o) str: Default is "Adam," could be "Adam," "SGD," or "Adagrad."

--weight_decay (-w) float: Default is 0.

An example command:
```
python dl.py -e example -m VGG -o SGD -b 16
```
