from functools import wraps
from django.shortcuts import redirect
from django.contrib import messages

def user_has_access(view_func):
    """
    Decorator to check if a user has access to the invoice reconciliation module.
    Redirects to the dashboard if access is denied.
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        # The access logic from the original `has_access` function.
        if not (request.user.is_superuser or hasattr(request.user, 'clientprofile')):
            messages.warning(request, "You do not have permission to access that page.")
            return redirect('dashboard')
        return view_func(request, *args, **kwargs)
    return _wrapped_view