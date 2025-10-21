from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("ml.urls")),
    path("api/", include("ml.api")),
]
