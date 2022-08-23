# ST-CLUSTERING

**st_clustering** is an open-source software package for spatial-temporal clustering:

- Built on top of `sklearn`'s clustering algorithms
- Scales to memory using chuncking. See the `st_fit_frame_split` method

## Installation
The easiest way to install *st_clustering* is by using `pip` :

    pip install st_clustering

## How to use

```python
import st_clustering as stc

st_dbscan = stc.ST_DBSCAN(eps1 = 0.05, eps2 = 10, min_samples = 5)
st_dbscan.st_fit(data)

```

- __Demo Notebook:__ this [Jupyter Notebook](/demo/demo.ipynb) shows a demo of common features in this package.

## Description

A package that implements a straightforward extension for various clustering algorithms to accomodate spatio-temporal data. 
Available algorithms are:

- ST DBSCAN
- ST Agglomerative
- ST OPTICS
- ST Spectral Clustering
- ST Affinity Propagation
- ST HDBSCAN

For more details please see original [paper](https://scibib.dbvis.de/uploadedFiles/Cakmak_ST_Clustering_Benchmark.pdf):

```
Cakmak, E., Plank, M., Calovi, D. S., Jordan, A., & Keim, D. (2021). Spatio-temporal clustering benchmark for collective animal behavior. In 1st ACM SIGSPATIAL International Workshop on Animal Movement Ecology and Human Mobility (HANIMOB’21).
```

## License
Released under MIT License. See the [LICENSE](LICENSE) file for details.
