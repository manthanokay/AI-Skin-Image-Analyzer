import unittest
import io
import tempfile
from unittest.mock import Mock, patch

import numpy as np

from app import app, ai_engine, processor
from backend.ai_analyzer import AIAnalyzer


class ChatEndpointTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.original_upload_folder = app.config["UPLOAD_FOLDER"]
        self.client = app.test_client()

    def tearDown(self):
        app.config["UPLOAD_FOLDER"] = self.original_upload_folder

    @patch.object(ai_engine, "chat", return_value="Try introducing one gentle product at a time.")
    def test_chat_returns_assistant_reply(self, chat):
        response = self.client.post("/api/chat", json={
            "message": "How should I start a routine?",
            "history": [],
            "skin_context": {"acne_like_regions": 2}
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["reply"], "Try introducing one gentle product at a time.")
        chat.assert_called_once()

    @patch.object(ai_engine, "chat")
    def test_chat_rejects_empty_messages(self, chat):
        response = self.client.post("/api/chat", json={"message": "  "})

        self.assertEqual(response.status_code, 400)
        chat.assert_not_called()

    @patch.object(processor, "preprocess")
    def test_analysis_rejects_images_without_a_detected_face(self, preprocess):
        preprocess.return_value = (
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((100, 100, 3), dtype=np.uint8),
            None
        )
        with tempfile.TemporaryDirectory() as upload_folder:
            app.config["UPLOAD_FOLDER"] = upload_folder
            response = self.client.post("/api/analyze", data={
                "image": (io.BytesIO(b"test image"), "photo.jpg")
            }, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 422)
        self.assertIn("No face detected", response.json["error"])

    def test_home_page_keeps_assistant_hidden_until_analysis_succeeds(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="chat-toggle" class="btn primary chat-toggle hidden"', response.data)
        self.assertIn(b'id="chat-panel"', response.data)

    @patch("backend.ai_analyzer.requests.post")
    def test_chat_completion_sends_bearer_authorization(self, post):
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}]
        }
        post.return_value = response
        analyzer = AIAnalyzer()
        analyzer.api_key = "test-key"
        analyzer.api_url = "https://example.test/chat/completions"

        result = analyzer.chat("Hello")

        self.assertEqual(result, "Hello!")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key")


if __name__ == "__main__":
    unittest.main()
