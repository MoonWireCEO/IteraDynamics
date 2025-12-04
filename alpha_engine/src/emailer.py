import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_alert(subject: str, body: str):
    sender_email = os.getenv("EMAIL_SENDER")  # e.g. your verified SendGrid sender
    receiver_email = os.getenv("EMAIL_RECEIVER")
    smtp_server = "smtp.sendgrid.net"
    smtp_port = 587
    smtp_username = "apikey"  # Always "apikey" when using SendGrid SMTP
    smtp_password = os.getenv("SENDGRID_API_KEY")  # Your actual API key

    if not all([sender_email, receiver_email, smtp_password]):
        raise EnvironmentError("Missing SendGrid environment variables.")
        print(f"SENDER={sender_email}")
        print(f"RECEIVER={receiver_email}")
        print(f"API_key={stmp_password}")

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
            print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")
