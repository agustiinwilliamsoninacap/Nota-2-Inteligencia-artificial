from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("insurance/", views.insurance_view, name="insurance"),
    path("diabetes/", views.diabetes_view, name="diabetes"),
    path("rf-report/", views.rf_report_view, name="rf_report"),
]
