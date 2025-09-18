from django.urls import path
from . import views
from django.views.generic.base import RedirectView

app_name = 'doc_generator'

urlpatterns = [
    # The single-page form view has been removed in favor of the multi-step wizard.
    path('<int:pk>/preview/', views.freshwater_plan_preview, name='freshwater_plan_preview'),

    # --- New Multi-Step Wizard URLs ---
    path('wizard/start/', views.plan_wizard_start, name='plan_wizard_start'),
    # The 'freshwater_plan_create' name is used in an older template.
    # Instead of serving the same view from two URLs, we'll permanently redirect
    # the old URL to the new, canonical one. This is better practice.
    path('freshwater/create/', RedirectView.as_view(pattern_name='doc_generator:plan_wizard_start', permanent=True), name='freshwater_plan_create'),
    path('wizard/<int:pk>/details/', views.plan_wizard_details, name='plan_wizard_details'),
    path('wizard/<int:pk>/status/', views.check_plan_status, name='check_plan_status'),
    path('wizard/<int:pk>/map-vulnerabilities/', views.plan_wizard_map_vulnerabilities, name='plan_wizard_map_vulnerabilities'),
    path('wizard/<int:pk>/map-activities/', views.plan_wizard_map_activities, name='plan_wizard_map_activities'),
    # Add more steps here in the future
    # path('wizard/<int:pk>/summary/', views.plan_wizard_summary, name='plan_wizard_summary'),
]