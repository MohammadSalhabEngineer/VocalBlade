import strawberry
import strawberry_django
from strawberry import Info
from gqlauth.core.utils import get_user
from gqlauth.user.queries import UserQueries
from gqlauth.core.types_ import GQLAuthError, GQLAuthErrors
from gqlauth.user import relay as mutations
from .types import User

@strawberry.type
class CoreQuery(UserQueries):
    @strawberry_django.field()
    def me(self, info: Info) -> User:
        user = get_user(info)
        if not user.is_authenticated:
            raise GQLAuthError(code=GQLAuthErrors.UNAUTHENTICATED)
        return user

@strawberry.type
class CoreMutation:
    
    verify_token = mutations.VerifyToken.field
    update_account = mutations.UpdateAccount.field
    archive_account = mutations.ArchiveAccount.field
    delete_account = mutations.DeleteAccount.field
    password_change = mutations.PasswordChange.field
    # swap_emails = mutations.SwapEmails.field
    # captcha = Captcha.field
    token_auth = mutations.ObtainJSONWebToken.field
    register = mutations.Register.field
    verify_account = mutations.VerifyAccount.field
    resend_activation_email = mutations.ResendActivationEmail.field
    send_password_reset_email = mutations.SendPasswordResetEmail.field
    password_reset = mutations.PasswordReset.field
    password_set = mutations.PasswordSet.field
    refresh_token = mutations.RefreshToken.field
    revoke_token = mutations.RevokeToken.field
    # verify_secondary_email = mutations.VerifySecondaryEmail.field

