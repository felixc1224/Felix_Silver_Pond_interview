# data/raw — untouched source data (not committed)

Put the official CIFAR-10 python archive here:

    data/raw/cifar-10-python.tar.gz

Source: https://www.cs.toronto.edu/~kriz/cifar.html  (Alex Krizhevsky, 2009)
MD5:    c58f30108f718f92721af3b95e74349a   (verify with `md5sum`)
Size:   170,498,071 bytes

`src/dataset.py` extracts it into `data/raw/cifar-10-batches-py/` on first run.
Nothing in this folder is edited by hand.
