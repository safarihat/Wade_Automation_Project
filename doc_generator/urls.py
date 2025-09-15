from django.urls import path
from .views import FreshwaterPlanCreateView, FreshwaterPlanPreviewView, maplibre_test_view

app_name = 'doc_generator'

urlpatterns = [
    path('freshwater-plan/create/', FreshwaterPlanCreateView.as_view(), name='freshwater_plan_create'),
    # Add a path for the preview page, which accepts the plan's ID
    path('freshwater-plan/<int:pk>/preview/', FreshwaterPlanPreviewView.as_view(), name='freshwater_plan_preview'),
    path('map-test/', maplibre_test_view, name='maplibre_test'),
]