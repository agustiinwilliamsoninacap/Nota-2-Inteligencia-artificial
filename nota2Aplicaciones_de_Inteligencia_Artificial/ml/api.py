from django.urls import path
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .services import predict_insurance, predict_diabetes

@api_view(["POST"])
def predict_insurance_api(request):
    return Response({"charges": predict_insurance(request.data)})

@api_view(["POST"])
def predict_diabetes_api(request):
    return Response(predict_diabetes(request.data))

urlpatterns = [
    path("predict/insurance", predict_insurance_api),
    path("predict/diabetes", predict_diabetes_api),
]
