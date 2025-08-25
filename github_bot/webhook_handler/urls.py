from django.urls import path
from . import views

urlpatterns = [
    path('', views.github_webhook, name='github_webhook'),  # Matches /webhook/
]
