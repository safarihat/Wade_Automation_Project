from django.urls import path
from .views import FreshwaterPlanCreateView

app_name = 'doc_generator'

urlpatterns = [
    path('freshwater-plan/create/', FreshwaterPlanCreateView.as_view(), name='freshwater_plan_create'),
]