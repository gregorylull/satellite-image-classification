
## Information theory
Reliably transmit msg from sender to receiver. Claude Shannon, reduce uncertainty by 2 with bits.

e.g.: If there's a 50:50 chance that is' rain or shine tomorrow, the weather channel can tell you it's "rainy" tomorrow, which is a lot of bit, but that information really is 1 bit of "useful" of information by reducing the uncertainty of the weather tomorrow. This is because these two outcomes are equally likely.

e.g. if there's 8 states of weather that's all equally likely, then the weather station can reduce your uncertainty from 8 to 1, which is 3 bits of "useful" information. 2^3 == 8


e.g. If there's a 75:25 that it's sunshine:rain tomorrow, the weather station by telling you that it'll rain has given your 2 bits of "useful" information because log2(4) == 2. The uncertainty reduction is the inverse of the probability, that is 1/0.25 (rain)

e.g. if there's a 99:1 sunshine:rain, then you inherently already feel certain that it'll be sunshine tomorrow. If the weather channel then says that it will be sunny, they haven't reduced your uncertainty by much at all. 


#### Sources
1. A short introduction to Entropy, Cross-Entropy and KL-Divergence 2018 [youtube](https://www.youtube.com/watch?v=ErfnhcEV1O8)




## Image detection and segmentation

Computationally more efficient to downsample and have deep layers, and then upsample (unpool) at the end.

### fixed unpooling
In network upsampling (unpooling), the following three methods are all "fixed", there's nothing being learned. The algorithm is discrete.

- nearest neighbor, 2x2 --> 4x4
```python
[
    [1, 2],
    [3, 4]
]

[
    [1, 1, 2, 2],
    [1, 1, 2, 2],
    [3, 3, 4, 4],
    [3, 3, 4, 4]
]
```

- "bed of nails" 2x2 --> 4x4
```python
[
    [1, 2],
    [3, 4]
]
[
    [1, 0, 2, 0],
    [0, 0, 0, 0],
    [3, 0, 4, 0],
    [0, 0, 0, 0]
]
```

- upsampling "max unpooling" with bed of nails. 
```python
# input with higher resolution
[
    [1, 3, 2, 1],
    [5, 4, 1, 1],
    [1, 3, 1, 2],
    [1, 2, 4, 2]
]

# max pooling and remembering which index
[
    [5, 2],
    [3, 4]
]

# upsampling and putting max in original index
[
    [0, 0, 2, 0],
    [5, 0, 0, 0],
    [0, 3, 0, 0],
    [0, 0, 4, 0]
]
```

### learnable upsampling, transpose convolution
- "deconvolution" is a bad term
- upconvolution 
- fractionally strided convolution
- backward strided convolution


#### Sources
1. Stanford - Lecture 11 Detection and Segmentation [youtube](https://www.youtube.com/watch?v=nDPWywWRIRo)


## Keras
### Conv2d Filters vs kernel size



- diff between the two [stackoverflow] (https://stackoverflow.com/questions/51180234/keras-conv2d-filters-vs-kernel-size)