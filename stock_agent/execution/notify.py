import os
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage

# Gmail requires an app password (Google Account -> Security -> App passwords).

DEFAULT_SMTP_HOST = "smtp.gmail.com"
DEFAULT_SMTP_PORT = 465
SMTP_TIMEOUT_SECONDS = 30


class NotifierNotConfigured(RuntimeError):
    pass


@dataclass(frozen=True)
class EmailNotifier:
    to: str
    user: str
    password: str = ""
    host: str = DEFAULT_SMTP_HOST
    port: int = DEFAULT_SMTP_PORT

    def __repr__(self) -> str:
        # Keeps the password out of tracebacks and logs.
        return f"EmailNotifier(to={self.to!r}, user={self.user!r}, host={self.host!r})"

    @classmethod
    def from_env(cls) -> "EmailNotifier":
        to = os.getenv("ALERT_EMAIL_TO", "")
        user = os.getenv("ALERT_SMTP_USER", "")
        password = os.getenv("ALERT_SMTP_PASSWORD", "")
        missing = [name for name, value in (("ALERT_EMAIL_TO", to),
                                            ("ALERT_SMTP_USER", user),
                                            ("ALERT_SMTP_PASSWORD", password)) if not value]
        if missing:
            raise NotifierNotConfigured(
                f"email alerting is not configured: set {', '.join(missing)} in the "
                "environment or the .env file at the repository root")
        return cls(to=to, user=user, password=password,
                   host=os.getenv("ALERT_SMTP_HOST", DEFAULT_SMTP_HOST),
                   port=int(os.getenv("ALERT_SMTP_PORT", str(DEFAULT_SMTP_PORT))))

    def send(self, subject: str, body: str) -> None:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.user
        message["To"] = self.to
        message.set_content(body)

        with smtplib.SMTP_SSL(self.host, self.port, timeout=SMTP_TIMEOUT_SECONDS,
                              context=ssl.create_default_context()) as smtp:
            smtp.login(self.user, self.password)
            smtp.send_message(message)
