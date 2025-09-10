from django.views.generic.edit import CreateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from .models import FreshwaterPlan
from .forms import FreshwaterPlanForm
# Assuming generate_plan_task is defined in doc_generator/tasks.py
from .tasks import generate_plan_task # This import assumes tasks.py exists

class FreshwaterPlanCreateView(LoginRequiredMixin, CreateView):
    model = FreshwaterPlan
    form_class = FreshwaterPlanForm
    template_name = 'doc_generator/freshwater_plan_form.html' # Assuming this template path
    success_url = reverse_lazy('freshwater_plan_preview') # Assuming this URL name

    def form_valid(self, form):
        # Set the user for the new FreshwaterPlan instance
        form.instance.user = self.request.user
        # Save the instance to get an ID before passing to Celery task
        self.object = form.save()
        # Call the Celery task to generate the plan in the background
        generate_plan_task.delay(self.object.pk)
        return super().form_valid(form)