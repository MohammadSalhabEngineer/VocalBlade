import strawberry
from strawberry_django.optimizer import DjangoOptimizerExtension
from core.schema import CoreQuery, CoreMutation
from transcriber.schema import TranscriberQuery, TranscriberMutation
from gqlauth.core.middlewares import JwtSchema

@strawberry.type
class Query(CoreQuery, TranscriberQuery): ...

@strawberry.type
class Mutation(CoreMutation, TranscriberMutation): ...


schema = JwtSchema(
    query=Query,
    mutation=Mutation,
    extensions=[
        DjangoOptimizerExtension,
    ],
)