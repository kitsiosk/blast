import json
import os
from django.test import TestCase
from .views import run  # Adjust import based on your app structure
from datetime import datetime

#
# RUN With: python manage.py test webhook_handler.tests.TestGenerationBugbug4867
#


class TestGenerationBugbug4867(TestCase):
    def setUp(self):
        # Load the local JSON file
        mock_payload = "test_mocks/bugbug_4867.json"
        payload_path = os.path.join(os.path.dirname(__file__), mock_payload)
        with open(payload_path, "r") as f:
            self.payload = json.load(f)
    
    def test_run_function_with_local_payload(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        response, _ = run(self.payload,
                    model="gpt-4o",
                    iAttempt=0,
                    timestamp=timestamp,
                    post_comment=False)

        
        self.assertIsNotNone(response)  # Ensure response is not None
        self.assertTrue(isinstance(response, dict) or hasattr(response, 'status_code'))  # Ensure response is a dict or HttpResponse