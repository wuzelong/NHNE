# NHNE 
Code for paper "Nuclear Norm-Based Network Embedding with High-order Proximity for Sparse Networks and Applications". PyTorch implementations of Nuclear norm Network Embedding (NNE) and Nuclear norm High-order Network Embedding (NHNE) algorithms are provided for downstream tasks such as network reconstruction, link prediction, node classification, and network visualization.

![framework](https://github.com/wuzelong/NHNE/assets/45838327/0e088a23-f8ec-4d9d-aa53-8dfc64dcb404)

*requirements*：
* torch
* numpy
* sklearn
* matplotlib
* liblinear-official
* iterative-stratification

*examples*：
1. Reconstruction
```bash
python reconstruction.py
```
2. Link Prediction
```bash
python linkPrediction.py
```
3. Node Classification
```bash
python classification.py
```
4. Network Visualization
```bash
python visualization.py
```

Tuning parameters based on the data set and task is required.
