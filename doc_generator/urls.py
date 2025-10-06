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
    path('api/get-parcel-geometry/', views.get_parcel_geometry, name='api_get_parcel_geometry'),
    path('api/nzlri-erosion/', views.NZLRIErosionLayerView.as_view(), name='api_nzlri_erosion'),
    path('api/protected-areas/', views.ProtectedAreasLayerView.as_view(), name='api_protected_areas'),
    path('api/groundwater-zones/', views.GroundwaterZonesLayerView.as_view(), name='api_groundwater_zones'),
    # The catchment area API endpoint for the map has been removed as per user request.
    path('wizard/<int:pk>/details/', views.plan_wizard_details, name='plan_wizard_details'),
    path('wizard/<int:pk>/status/', views.check_plan_status, name='api_check_plan_status'),
    path('api/vulnerability-analysis/<int:pk>/', views.api_generate_vulnerability_analysis, name='api_generate_vulnerability_analysis'), # This was already correct, but ensuring consistency.
    path('wizard/<int:pk>/map-vulnerabilities/', views.plan_wizard_map_vulnerabilities, name='plan_wizard_map_vulnerabilities'),
    path('wizard/<int:pk>/map-activities/', views.plan_wizard_map_activities, name='plan_wizard_map_activities'),
    path('wizard/<int:pk>/map-works/', views.plan_wizard_map_works, name='plan_wizard_map_works'),
    path('wizard/<int:pk>/risk-management/', views.plan_wizard_risk_management, name='plan_wizard_risk_management'),

    # New API endpoint for the modular risk analysis pipeline
    path('api/risk-analysis/<int:pk>/', views.api_generate_risk_report, name='api_generate_risk_report'),

    # V2 endpoint for the improved vulnerability analysis pipeline
    path('api/vulnerability-analysis/v2/<int:pk>/', views.api_generate_vulnerability_analysis_v2, name='api_generate_vulnerability_analysis_v2'),

    # API for fetching water quality data
    path('api/get-water-quality-data/', views.api_get_water_quality_data, name='api_get_water_quality_data'),

    # Add more steps here in the future
    # path('wizard/<int:pk>/summary/', views.plan_wizard_summary, name='plan_wizard_summary'),
]