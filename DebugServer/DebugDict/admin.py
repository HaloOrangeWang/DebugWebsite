from .models import ViewCount
from django.contrib import admin

# Register your models here.


class ViewCountAdmin(admin.ModelAdmin):
    list_per_page = 20
    readonly_fields = ('date2', 'view_count')
    fields = ('date2', 'view_count')


admin.site.register(ViewCount, ViewCountAdmin)
