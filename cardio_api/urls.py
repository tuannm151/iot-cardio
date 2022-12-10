from django.urls import path, include
from .views import (CardioMLView)

urlpatterns = [
    path('api', CardioMLView.as_view()),
]
