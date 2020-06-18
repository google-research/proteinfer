# ProteInfer

ProteInfer is an approach for predicting the functional properties of protein
sequences using deep neural networks.

To install:
```
pip3 install -q -r requirements.txt
```
If running on macOS you will need to remove the `tensorflow-gpu` line from requirements.txt and GPU-acceleration will not be available.

To run the unit tests:
```
bash -c 'for f in *_test.py; do python3 $f || exit 1; done'
```

## Status

This repository is still a work in progress. Please check back for more
documentation and a manuscript very soon. We are not able to accept pull 
requests or contributions at this time.
