from django.contrib.auth.models import User
from django.db import models

class ClientProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    company_name = models.CharField(max_length=100)
    dashboard_data = models.TextField(blank=True)

    def __str__(self):
        return self.user.username
