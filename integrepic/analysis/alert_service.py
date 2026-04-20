"""
Alert Service - Pushes critical detection events to external Webhooks (Discord, Slack, etc.)
"""
import requests
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

class AlertService:
    @staticmethod
    def trigger_deepfake_alert(analysis, webhook_url=None):
        """Send a notification if high probability deepfake is detected"""
        if not webhook_url:
            # Fallback to local setting or do nothing if none exists
            webhook_url = getattr(settings, 'DISCORD_WEBHOOK_URL', '')
            
        if not webhook_url:
            logger.info("No webhook URL configured. Alert skipped.")
            return False

        try:
            payload = {
                "username": "IntegriPic Engine",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/8662/8662283.png",
                "embeds": [
                    {
                        "title": "🚨 High-Probability Deepfake Detected",
                        "color": 15158332,  # Red
                        "description": f"An uploaded asset triggered the AI Detection heuristic.",
                        "fields": [
                            {"name": "Filename", "value": analysis.original_filename, "inline": True},
                            {"name": "Confidence", "value": f"{analysis.deepfake_probability}%", "inline": True},
                            {"name": "Heuristic Notes", "value": analysis.deepfake_notes or "N/A", "inline": False},
                            {"name": "User", "value": analysis.user.username if analysis.user else "Anonymous", "inline": True}
                        ]
                    }
                ]
            }
            
            # Send payload async or quickly via requests checkout
            # Note: In production this should be a Celery task.
            resp = requests.post(webhook_url, json=payload, timeout=3)
            return resp.status_code == 204
        except Exception as e:
            logger.error(f"Failed to push alert: {e}")
            return False
