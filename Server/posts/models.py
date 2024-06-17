from django.db import models

# Create your models here.

class Post (models.Model):

    image = models.ImageField(verbose_name = '이미지')