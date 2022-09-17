# HPOBench exclusively for the newly added tabular benchmarks

[HPOBench](https://github.com/automl/HPOBench) is a collection of hyperparameter optimization benchmarks.

In this repository, we provide the code to use the MLP tabular benchmark newly added in the paper [`HPOBench: A Collection of Reproducible Multi-Fidelity Benchmark Problems for HPO`](https://arxiv.org/abs/2109.06716).

# Usage

1. Clone this repository:
```
$ git clone https://github.com/nabenabe0928/easy-hpo-bench.git
```

2. Install the requirements
```
$ pip install -r requirements.txt
```

3. Download the dataset from the following URLs:
    * [MLP](https://ndownloader.figshare.com/files/30379005) (372MB)
    * [XGBoost](https://ndownloader.figshare.com/files/30469920) (258MB)
    * [Random Forest](https://ndownloader.figshare.com/files/30469089) (406MB)

4. Test by the following command:
```
python example.py
```
