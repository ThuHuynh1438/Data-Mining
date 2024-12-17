import base64
from django import template

register = template.Library()

@register.filter(name='base64encode')
def base64encode(value):
    """
    Custom template filter to encode a string or bytes to base64.
    """
    if isinstance(value, bytes):
        encoded = base64.b64encode(value).decode('utf-8')
    else:
        encoded = base64.b64encode(value.encode('utf-8')).decode('utf-8')
    return encoded
