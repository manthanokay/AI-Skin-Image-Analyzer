import os

import requests
from dotenv import load_dotenv

load_dotenv()


class AIServiceError(Exception):
    pass


class AIAnalyzer:
    def __init__(self):
        self.api_key = os.getenv("AI_API_KEY")
        self.api_url = os.getenv("AI_API_URL")
        self.model = os.getenv("AI_MODEL", "gpt-4o-mini")

    def _request_completion(self, messages, temperature=0.7):
        if not self.api_key or self.api_key == "your_api_key_here" or not self.api_url:
            raise AIServiceError("AI service is not configured. Set AI_API_KEY and AI_API_URL.")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise AIServiceError("AI service is unavailable. Please try again later.") from exc

        try:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise AIServiceError("AI service returned an invalid response.") from exc
        if not isinstance(content, str) or not content.strip():
            raise AIServiceError("AI service returned an empty response.")
        return content.strip()

    def analyze_skin_with_ai(self, dip_results):
        """Send image-analysis metrics for an educational interpretation."""
        if not self.api_key or self.api_key == "your_api_key_here" or not self.api_url:
            return {
                "error": "AI service is not configured",
                "interpretation": "Local analysis complete. AI interpretation is unavailable because the AI service is not configured."
            }

        prompt = (
            "Interpret these computer-vision estimates in gentle, educational language. "
            "Do not diagnose; explain that the detected characteristics are estimates and "
            "suggest when professional evaluation may be appropriate.\n\n"
            f"Redness score: {dip_results.get('redness_score')}\n"
            f"Dark spots: {dip_results.get('dark_spots')}\n"
            f"Pimple-like regions: {dip_results.get('acne_like_regions')}\n"
            f"Pore-like regions: {dip_results.get('pore_like_regions')}\n"
            f"Skin smoothness: {dip_results.get('skin_smoothness')}%\n"
            f"Image quality: {dip_results.get('image_quality')}/100"
        )

        try:
            interpretation = self._request_completion([
                {
                    "role": "system",
                    "content": (
                        "You are a professional skin education assistant. Never provide "
                        "medical diagnoses; describe visual characteristics only."
                    )
                },
                {"role": "user", "content": prompt}
            ])
            return {"interpretation": interpretation}
        except AIServiceError as exc:
            return {
                "error": str(exc),
                "interpretation": "AI interpretation is currently unavailable. Please refer to the local image processing results."
            }

    def chat(self, message, history=None, skin_context=None):
        """Answer a user question, optionally using the latest analysis estimates."""
        messages = [{
            "role": "system",
            "content": (
                "You are LuminaSkin, a helpful, friendly assistant. Answer questions clearly, "
                "including general questions and skincare questions. For skincare, give practical, "
                "balanced educational information, mention patch testing and professional care "
                "when appropriate, and never diagnose a condition or prescribe treatment. Image "
                "analysis counts are computer-vision estimates, not confirmed diagnoses. Use any "
                "provided analysis context with that uncertainty, and do not claim to see an image "
                "unless the user has shared one."
            )
        }]
        if skin_context:
            messages.append({
                "role": "system",
                "content": f"Current image-analysis estimates (not a diagnosis): {skin_context}"
            })
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})
        return self._request_completion(messages, temperature=0.5)
