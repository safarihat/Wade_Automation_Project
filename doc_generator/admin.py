from django.contrib.gis import admin
from .models import FreshwaterPlan

@admin.register(FreshwaterPlan)
class FreshwaterPlanAdmin(admin.GISModelAdmin):
    """
    Admin interface for the FreshwaterPlan model.
    """
    list_display = ('council', 'user', 'latitude', 'longitude', 'payment_status', 'created_at')
    list_filter = ('payment_status', 'council', 'created_at')
    search_fields = ('council', 'user__username', 'generated_plan')
    readonly_fields = ('created_at', 'updated_at')
    fieldsets = (
        (None, {
            'fields': ('user', 'council', 'payment_status')
        }),
        ('Location Information', {
            'fields': ('latitude', 'longitude', 'location', 'map_image')
        }),
        ('Plan Documents', {
            'fields': ('generated_plan', 'pdf_preview', 'pdf_final')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        }),
    )