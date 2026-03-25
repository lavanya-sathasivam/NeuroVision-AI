import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json
from pathlib import Path

NOTIFICATIONS_LOG = Path("notifications_log.json")


def _load_notifications_log():
    """Load notification history."""
    if not NOTIFICATIONS_LOG.exists():
        return []
    try:
        with open(NOTIFICATIONS_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_notifications_log(logs):
    """Save notification history."""
    with open(NOTIFICATIONS_LOG, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)


def send_email(
    sender_email,
    sender_password,
    recipient_email,
    subject,
    body,
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    is_html=False
):
    """
    Send email notification.
    
    Args:
        sender_email: Sender email address
        sender_password: Sender email password or app password
        recipient_email: Recipient email address (string or list)
        subject: Email subject
        body: Email body content
        smtp_server: SMTP server address
        smtp_port: SMTP port
        is_html: Whether body is HTML
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender_email
        
        # Handle multiple recipients
        if isinstance(recipient_email, list):
            msg["To"] = ", ".join(recipient_email)
            recipients = recipient_email
        else:
            msg["To"] = recipient_email
            recipients = [recipient_email]
        
        # Attach body
        if is_html:
            msg.attach(MIMEText(body, "html"))
        else:
            msg.attach(MIMEText(body, "plain"))
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipients, msg.as_string())
        
        # Log notification
        _log_notification(
            sender_email, 
            recipients, 
            subject, 
            "email", 
            success=True
        )
        
        return True, "Email sent successfully"
    
    except smtplib.auth.AuthenticationError:
        msg = "Authentication failed. Check email and password."
        _log_notification(
            sender_email, 
            recipients if 'recipients' in locals() else [], 
            subject, 
            "email", 
            success=False, 
            error=msg
        )
        return False, msg
    
    except smtplib.SMTPException as e:
        msg = f"SMTP error: {str(e)}"
        _log_notification(
            sender_email, 
            recipients if 'recipients' in locals() else [], 
            subject, 
            "email", 
            success=False, 
            error=msg
        )
        return False, msg
    
    except Exception as e:
        msg = f"Error sending email: {str(e)}"
        _log_notification(
            sender_email, 
            recipients if 'recipients' in locals() else [], 
            subject, 
            "email", 
            success=False, 
            error=msg
        )
        return False, msg


def _log_notification(sender, recipients, subject, notif_type, success=True, error=None):
    """Log notification attempt."""
    logs = _load_notifications_log()
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "sender": sender,
        "recipients": recipients if isinstance(recipients, list) else [recipients],
        "subject": subject,
        "type": notif_type,
        "success": success,
        "error": error
    }
    
    logs.append(log_entry)
    _save_notifications_log(logs)


def get_notification_history(limit=50):
    """Get recent notification history."""
    logs = _load_notifications_log()
    return logs[-limit:]


def generate_scan_report_email(patient_name, scan_prediction, confidence, decision, probabilities, doctor_name):
    """Generate HTML email for scan report."""
    html_body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #2c3e50;">🧠 NeuroVision AI - Scan Report</h2>
                <hr style="border: none; border-top: 2px solid #3498db;">
                
                <div style="background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Doctor:</strong> {doctor_name}</p>
                    <p><strong>Patient:</strong> {patient_name}</p>
                </div>
                
                <h3 style="color: #2c3e50;">Analysis Results:</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background-color: #3498db; color: white;">
                        <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Prediction</strong></td>
                        <td style="padding: 10px; border: 1px solid #bdc3c7;">{scan_prediction.capitalize()}</td>
                    </tr>
                    <tr style="background-color: #ecf0f1;">
                        <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Confidence</strong></td>
                        <td style="padding: 10px; border: 1px solid #bdc3c7;">{confidence*100:.1f}%</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>Decision</strong></td>
                        <td style="padding: 10px; border: 1px solid #bdc3c7;"><strong>{decision}</strong></td>
                    </tr>
                </table>
                
                <h3 style="color: #2c3e50;">Probability Distribution:</h3>
                <ul>
    """
    
    for class_name, prob in probabilities.items():
        html_body += f"<li><strong>{class_name}:</strong> {prob*100:.1f}%</li>"
    
    html_body += """
                </ul>
                
                <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0;">
                    <strong>⚠️ Disclaimer:</strong> This is AI-assisted analysis. Final diagnosis must be made by a qualified radiologist.
                </div>
                
                <hr style="border: none; border-top: 2px solid #3498db;">
                <p style="text-align: center; color: #7f8c8d; font-size: 12px;">
                    NeuroVision AI v2.0 - Clinical Decision Support System
                </p>
            </div>
        </body>
    </html>
    """
    
    return html_body


def generate_summary_report_email(doctor_name, total_scans, predictions_summary, accuracy_rate):
    """Generate HTML email for summary report."""
    html_body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #2c3e50;">📊 NeuroVision AI - Weekly Summary Report</h2>
                <hr style="border: none; border-top: 2px solid #3498db;">
                
                <div style="background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <p><strong>Report Date:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
                    <p><strong>Doctor:</strong> {doctor_name}</p>
                </div>
                
                <h3 style="color: #2c3e50;">📈 Analysis Summary:</h3>
                <div style="background-color: #d5f4e6; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <p><strong>Total Scans Analyzed:</strong> {total_scans}</p>
                    <p><strong>Model Confidence Rate:</strong> {accuracy_rate:.1f}%</p>
                </div>
                
                <h3 style="color: #2c3e50;">📋 Prediction Breakdown:</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background-color: #3498db; color: white;">
                        <th style="padding: 10px; border: 1px solid #bdc3c7;">Class</th>
                        <th style="padding: 10px; border: 1px solid #bdc3c7;">Count</th>
                        <th style="padding: 10px; border: 1px solid #bdc3c7;">Percentage</th>
                    </tr>
    """
    
    if predictions_summary:
        total = sum(predictions_summary.values())
        for class_name, count in predictions_summary.items():
            percentage = (count / total * 100) if total > 0 else 0
            html_body += f"""
                    <tr style="background-color: #ecf0f1;">
                        <td style="padding: 10px; border: 1px solid #bdc3c7;">{class_name}</td>
                        <td style="padding: 10px; border: 1px solid #bdc3c7;">{count}</td>
                        <td style="padding: 10px; border: 1px solid #bdc3c7;">{percentage:.1f}%</td>
                    </tr>
            """
    
    html_body += """
                </table>
                
                <hr style="border: none; border-top: 2px solid #3498db;">
                <p style="text-align: center; color: #7f8c8d; font-size: 12px;">
                    NeuroVision AI v2.0 - Clinical Decision Support System
                </p>
            </div>
        </body>
    </html>
    """
    
    return html_body


def test_email_configuration(sender_email, sender_password, smtp_server="smtp.gmail.com", smtp_port=587):
    """Test email configuration."""
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
        return True, "Email configuration is valid"
    except smtplib.auth.AuthenticationError:
        return False, "Authentication failed. Check email and password."
    except Exception as e:
        return False, f"Connection error: {str(e)}"
