# RepVGG-caffe

This is a caffe demo for [RepVGG](https://github.com/DingXiaoH/RepVGG)
you can download converted models from [releases](https://github.com/imistyrain/RepVGG-caffe/releases)
we also provide [train-mode models](models/RepVGG-A0-train.prototxt) as if you'd like to train it by caffe.

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