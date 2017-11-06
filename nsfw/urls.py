from django.conf.urls import url
from . import views

nsfw = views.NSFW()

urlpatterns = [
    url(r'^$', nsfw.post_list, name='home'),
    url(r'^origin/$', nsfw.origin, name='origin'),
    url(r'^convert/$', nsfw.convert, name='convert'),
    ]
