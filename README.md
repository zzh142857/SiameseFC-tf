# SiamFC-tf
This repository is the tensorflow implementation of both training and evaluation of SiamFC described in the paper [*Fully-Convolutional Siamese nets for object tracking*](https://www.robots.ox.ac.uk/~luca/siamese-fc.html).   
The code is revised on the base of the evaluation-only version code from the repository of the auther of this paper(https://github.com/torrvision/siamfc-tf).

## Preparation
1) Clone the repository
`git clone https://github.com/zzh142857/SiameseFC-tf.git`
2)The code is prepared for a environment with Python==3.6 and Tensorflow-gpu==1.6 with the required CUDA and CudNN library.   
3)Other packages can be installed by:   
`sudo pip install -r requirements.txt`
4)Download training data
cd to the directory where this README.md file located, then:
`mkdir data`
Download [video sequences](https://drive.google.com/file/d/0B7Awq_aAemXQSnhBVW5LNmNvUU0/view) in `data` and unzip the archive.


## Running the tracker
1) Set `video` from `parameters.evaluation` to `"all"` or to a specific sequence (e.g. `"vot2016_ball1"`)
1) See if you are happy with the default parameters in `parameters/hyperparameters.json`
1) Optionally enable visualization in `parameters/run.json`
1) Call the main script (within an active virtualenv session)
`python run_tracker_evaluation.py`

## Author
**Zhenghao Zhao**

## Authors of the original paper and code
* [**Luca Bertinetto**](https://www.robots.ox.ac.uk/~luca)
* [**Jack Valmadre**](http://jack.valmadre.net)

## References
If you find our work useful, please consider citing the original authors

↓ [Original method] ↓
```
@inproceedings{bertinetto2016fully,
  title={Fully-Convolutional Siamese Networks for Object Tracking},
  author={Bertinetto, Luca and Valmadre, Jack and Henriques, Jo{\~a}o F and Vedaldi, Andrea and Torr, Philip H S},
  booktitle={ECCV 2016 Workshops},
  pages={850--865},
  year={2016}
}
```
↓ [Improved method and evaluation] ↓
```
@article{valmadre2017end,
  title={End-to-end representation learning for Correlation Filter based tracking},
  author={Valmadre, Jack and Bertinetto, Luca and Henriques, Jo{\~a}o F and Vedaldi, Andrea and Torr, Philip HS},
  journal={arXiv preprint arXiv:1704.06036},
  year={2017}
}
```

## License
This code can be freely used for personal, academic, or educational purposes.
Please contact us for commercial use.

