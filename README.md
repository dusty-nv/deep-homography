### Deep Homography

A homography dataset can be constructed by taking a patch from an image and perturbing the corners. The 4 point correspondences between the original patch and the perturbed ones define a homography. A neural network can learn to produce the parameterized homography given the patches from the warped and unwarped image. 

This repo reproduces the work of [DeTone et al. 2016](https://arxiv.org/pdf/1606.03798.pdf). Check out my [blog post](https://ekrim.github.io/computer/vision,pytorch,homography/2018/08/07/deep-homography-estimation.html) for more analysis.

#### Making the data

Download the unlabeled [MS-COCO](http://cocodataset.org/#home) dataset and unzip such that the ~330k jpeg images are in `data/unlabeled2017`.

Build the data generation code that uses OpenCV:

```
cd data
mkdir build
cd build
cmake ..
make
```

Then run the data generation program:

```
./make_homography_data ../unlabeled2017 <int_patch_size> <int_max_jitter> <n_samples>
```

Use a patch size of 128, a max jitter of 32, and 500,000 samples to reproduce my results.

To train the network (saves each epoch):

```
python main.py
```

To see the performance on, e.g., the 4th image:

```
python eval.py --i 3
```

#### Eval Notes

If during `eval.py` you recieve the error `TypeError: Couldn't find conversion for foreign struct 'cairo.Context'`, install the following:

```
$ sudo apt-get install python-gi-cairo
```

Other dependencies of `eval.py`:

```
$ pip install matplotlib
```

#### ONNX Export

To export an ONNX model of the network from PyTorch to `deep_homography.onnx`:

```
$ python onnx_export.py
```

If you want to validate that the exported ONNX model is well-formed, first install these packages:

```
$ sudo apt-get install protobuf-compiler libprotoc-dev
$ pip install onnx --user
```

Then run the validation script:

```
$ python onnx_validate.py
```



 
