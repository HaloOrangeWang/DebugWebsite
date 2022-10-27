from django.db import models

# Create your models here.


class ViewCount(models.Model):
    date2 = models.IntegerField("日期", unique=True, primary_key=True)
    view_count = models.IntegerField("访问量")
