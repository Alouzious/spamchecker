from django.urls import path
from . import views

urlpatterns = [
    path('', views.classify_message, name='classify_message'),  # root of the app
]