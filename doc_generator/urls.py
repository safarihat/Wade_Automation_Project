from django.urls import path
from . import views

app_name = 'doc_generator'

urlpatterns = [
    # Add an alias for the old URL name to fix the NoReverseMatch error from the home template.
    path('plan/create/', views.plan_wizard_start, name='freshwater_plan_create'),

    # Wizard Steps
    path('plan/start/', views.plan_wizard_start, name='plan_wizard_start'),
    path('plan/<int:pk>/details/', views.plan_wizard_details, name='plan_wizard_details'),
    path('plan/<int:pk>/vulnerabilities/', views.plan_wizard_map_vulnerabilities, name='plan_wizard_map_vulnerabilities'),
    path('plan/<int:pk>/activities/', views.plan_wizard_map_activities, name='plan_wizard_map_activities'),
    path('plan/<int:pk>/works/', views.plan_wizard_map_works, name='plan_wizard_map_works'),
    path('plan/<int:pk>/risk-management/', views.plan_wizard_risk_management, name='plan_wizard_risk_management'),
    path('plan/<int:pk>/preview/', views.freshwater_plan_preview, name='freshwater_plan_preview'),
    path('plan/<int:pk>/water-quality/', views.water_quality_summary, name='water_quality_summary'),

    # API Endpoints
    path('api/check-plan-status/<int:pk>/', views.check_plan_status, name='check_plan_status'),
    path('api/get-parcel-geometry/', views.get_parcel_geometry, name='api_get_parcel_geometry'),
    path('api/nzlri-erosion/', views.NZLRIErosionLayerView.as_view(), name='api_nzlri_erosion'),
    path('api/protected-areas/', views.ProtectedAreasLayerView.as_view(), name='api_protected_areas'),
    path('api/groundwater-zones/', views.GroundwaterZonesLayerView.as_view(), name='api_groundwater_zones'),
    path('api/vulnerability-analysis/v2/<int:pk>/', views.api_generate_vulnerability_analysis_v2, name='api_generate_vulnerability_analysis_v2'),
    path('api/vulnerability-analysis/v2/<int:pk>/status/', views.api_check_vulnerability_analysis_status, name='api_check_vulnerability_analysis_status'),
    path('api/get-lawa-sites-data/', views.api_get_closest_lawa_sites_data, name='api_get_closest_lawa_sites_data'),
    path('api/check-job-status/', views.check_job_status, name='check_job_status'),
    
    path('ask/', views.ask_question_view, name='ask_question'),
]