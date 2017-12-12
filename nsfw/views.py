from django.shortcuts import render, HttpResponse, render_to_response
from django.template import RequestContext
from django.views.decorators.csrf import csrf_protect
from django.core.cache import cache
import chainer
from chainer import functions as F
from chainer.links import caffe
from PIL import Image
import numpy as np


# Create your views here.
class NSFW(object):
    def __init__(self):
        self.response_org = None
        self.response_scr = None
        self.model_cache_key = 'nsfw'

    def post_list(self, request):
        if request.method == 'POST':
            model = cache.get(self.model_cache_key)
            if model is None:
                model = caffe.CaffeFunction(
                    'nsfw_model/resnet_50_1by2_nsfw.caffemodel')
                cache.set(self.model_cache_key, model, None)
            img_org = Image.open(request.FILES['file'])
            if img_org.mode != "RGB":
                img_org = img_org.convert('RGB')

            img = np.asarray(img_org).astype(np.float32) / 255
            h, w, _ = img.shape
            img = np.expand_dims(img.transpose((2, 0, 1)), 0)

            with chainer.using_config('use_cudnn', 'never'):
                x, = model(
                    inputs={'data': F.resize_images(img, (224, 224))},
                    outputs=['eltwise_stage3_block2'],
                    disable=['pool', 'fc_nsfw', 'prob'])

            W = model.fc_nsfw.W.data.reshape(2, 1024, 1, 1)
            b = model.fc_nsfw.b.data
            output = chainer.functions.softmax(
                chainer.functions.convolution_2d(x, W, b))
            score = chainer.functions.resize_images(output, (h, w))[0, 1:]
            score = chainer.functions.resize_images(output, (h, w))[0, :1]
            img_scr = np.asarray(
                np.clip((img[0] * score.data).transpose((1, 2, 0)) * 255,
                        0, 255), dtype=np.uint8)
            self.response_org = HttpResponse(content_type='image/png')
            self.response_scr = HttpResponse(content_type='image/png')
            img_org.save(self.response_org, 'PNG')
            Image.fromarray(img_scr).save(self.response_scr, 'PNG')
        else:
            self.response_org = None
            self.response_scr = None
        return render(request, 'index.html', {})

    def convert(self, request):
        return self.response_scr

    def origin(self, request):
        return self.response_org
