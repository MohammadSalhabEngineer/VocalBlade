from . import models
from strawberry import auto
import strawberry_django

@strawberry_django.type(models.User)
class User:
    username: auto