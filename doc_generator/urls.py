from django.urls import path
from django.views.generic import TemplateView
from .views import FreshwaterPlanCreateView

app_name = 'doc_generator'

urlpatterns = [
    path('create/', FreshwaterPlanCreateView.as_view(), name='freshwater_plan_create'),
    path('preview/', TemplateView.as_view(template_name='doc_generator/freshwater_plan_preview.html'), name='freshwater_plan_preview'),
]
