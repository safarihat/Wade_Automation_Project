from django.urls import path, include
from django.contrib.auth import views as auth_views
from . import views
from .views import customer_dashboard

urlpatterns = [
    path('', views.home, name='home'),  # Root URL
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('pathways/', views.pathways_view, name='pathways'),
    path('dashboard/', customer_dashboard, name='dashboard'),
    

    # Authentication URLs
    path('login/', auth_views.LoginView.as_view(template_name='wade_automation/registration/login.html'), name='login'),
    path('logout/', views.logout_view, name='logout'),
]
