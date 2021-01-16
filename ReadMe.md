# RepVGG-caffe

This is a caffe demo for [RepVGG](https://github.com/DingXiaoH/RepVGG), 
you can download converted models from [releases](https://github.com/imistyrain/RepVGG-caffe/releases), 
we also provide [train-mode models](models/RepVGG-A0-train.prototxt) as if you'd like to train it by caffe.

![](https://pic3.zhimg.com/v2-bb6b4a145d239937c9a3c4e03dbd2199_1440w.jpg?source=172ae18bG)

## FLOPs

```
RepVGG-A0.prototxt
layer name Filter Shape     Output Size      Params   Flops        Ratio
conv1     (48, 3, 3, 3)    (1, 48, 112, 112) 1296     16257024     1.194%
conv2     (48, 48, 3, 3)   (1, 48, 56, 56)  20736    65028096     4.776%
conv3     (48, 48, 3, 3)   (1, 48, 56, 56)  20736    65028096     4.776%
conv4     (96, 48, 3, 3)   (1, 96, 28, 28)  41472    32514048     2.388%
conv5     (96, 96, 3, 3)   (1, 96, 28, 28)  82944    65028096     4.776%
conv6     (96, 96, 3, 3)   (1, 96, 28, 28)  82944    65028096     4.776%
conv7     (96, 96, 3, 3)   (1, 96, 28, 28)  82944    65028096     4.776%
conv8     (192, 96, 3, 3)  (1, 192, 14, 14) 165888   32514048     2.388%
conv9     (192, 192, 3, 3) (1, 192, 14, 14) 331776   65028096     4.776%
conv10    (192, 192, 3, 3) (1, 192, 14, 14) 331776   65028096     4.776%
conv11    (192, 192, 3, 3) (1, 192, 14, 14) 331776   65028096     4.776%
conv12    (192, 192, 3, 3) (1, 192, 14, 14) 331776   65028096     4.776%
conv13    (192, 192, 3, 3) (1, 192, 14, 14) 331776   65028096     4.776%
conv14    (192, 192, 3, 3) (1, 192, 14, 14) 331776   65028096     4.776%
conv15    (192, 192, 3, 3) (1, 192, 14, 14) 331776   65028096     4.776%
conv16    (192, 192, 3, 3) (1, 192, 14, 14) 331776   65028096     4.776%
conv17    (192, 192, 3, 3) (1, 192, 14, 14) 331776   65028096     4.776%
conv18    (192, 192, 3, 3) (1, 192, 14, 14) 331776   65028096     4.776%
conv19    (192, 192, 3, 3) (1, 192, 14, 14) 331776   65028096     4.776%
conv20    (192, 192, 3, 3) (1, 192, 14, 14) 331776   65028096     4.776%
conv21    (192, 192, 3, 3) (1, 192, 14, 14) 331776   65028096     4.776%
conv22    (1280, 192, 3, 3) (1, 1280, 7, 7)  2211840  108380160    7.961%
fc1       (1000, 1280)     (1, 1000)        1280000  1280000      0.094%
Layers num: 23
Total number of parameters:  8303888
Total number of FLOPs:  1361451008
```

## demo
```
python demo.py
```

sample outputs:
```
demo caffe
282 n02123159 tiger cat 0.29690439
281 n02123045 tabby, tabby cat 0.14270334
285 n02124075 Egyptian cat 0.12931268
263 n02113023 Pembroke, Pembroke Welsh corgi 0.10508225
278 n02119789 kit fox, Vulpes macrotis 0.046900906

demo dnn
282 n02123159 tiger cat 0.2969048
281 n02123045 tabby, tabby cat 0.142703
285 n02124075 Egyptian cat 0.12931274
263 n02113023 Pembroke, Pembroke Welsh corgi 0.10508249
278 n02119789 kit fox, Vulpes macrotis 0.046901245

demo pytorch
282 n02123159 tiger cat 0.29690438508987427
281 n02123045 tabby, tabby cat 0.14270293712615967
285 n02124075 Egyptian cat 0.1293124407529831
263 n02113023 Pembroke, Pembroke Welsh corgi 0.10508245974779129
278 n02119789 kit fox, Vulpes macrotis 0.04690106585621834
```