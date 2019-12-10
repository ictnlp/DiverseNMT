# Modeling Fluency and Faithfulness for Diverse Neural Machine Translation

PyTorch implementation of the models described in the AAAI2020 long paper [Modeling Fluency and Faithfulness for Diverse Neural Machine Translation](https://arxiv.org/pdf/1912.00178.pdf).

## Related code

Implemented based on [Fairseq-py](https://github.com/pytorch/fairseq), an open-source toolkit released by Facebook which was implemented strictly referring to [Vaswani et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf).

## DiverseNMT

Download [the preprocessed WMT'16 EN-DE data provided by Google](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) and extract it and preprocess the dataset with a joined dictionary.

Train a base model.

```
$ sh run_ende.sh
```

Inference the test set.

```
$ python generate.py $DATA --path $SMODEL \
    --gen-subset test --beam 4 --batch-size 128 \
    --remove-bpe --lenpen 0.6 > pred.de \
# because fairseq's output is unordered, we need to recover its order
$ grep ^H pred.de | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.de
```

## Citation
If you find the resources in this repository useful, please consider citing:

```
@article{feng2019modeling,
  title={Modeling Fluency and Faithfulness for Diverse Neural Machine Translation},
  author={Feng, Yang and Xie, Wanying and Gu, Shuhao and Shao, Chenze and Zhang, Wen and Yang, Zhengxin and Yu, Dong},
  journal={arXiv preprint arXiv:1912.00178},
  year={2019}
}
```
