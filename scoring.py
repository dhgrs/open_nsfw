import sys

import chainer
from chainer.links import caffe
from PIL import Image
import numpy as np


def score(path):
    model = caffe.CaffeFunction('nsfw_model/resnet_50_1by2_nsfw.caffemodel')

    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert('RGB')

    img = np.asarray(img).astype(np.float32) / 255
    h, w, _ = img.shape
    img = np.expand_dims(img.transpose((2, 0, 1)), 0)
    with chainer.using_config('use_cudnn', 'never'):
        x, = model(inputs={'data': img}, outputs=['eltwise_stage3_block2'],
                   disable=['pool', 'fc_nsfw', 'prob'])

    W = model.fc_nsfw.W.data.reshape(2, 1024, 1, 1)
    b = model.fc_nsfw.b.data
    output = chainer.functions.softmax(
        chainer.functions.convolution_2d(x, W, b))
    score = chainer.functions.resize_images(output, (h, w))[0, 1:]
    score = chainer.functions.resize_images(output, (h, w))[0, :1]
    scored_img = np.asarray(
        np.clip((img[0] * score.data).transpose((1, 2, 0)) * 255, 0, 255),
        dtype=np.uint8)
    Image.fromarray(scored_img).save('result.png')


score(sys.argv[1])
