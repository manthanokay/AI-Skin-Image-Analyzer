import json

class ProductRecommender:
    def __init__(self):
        self.product_db = [
            {"name": "Gentle Cleanser", "category": "cleanser", "for": ["acne_like", "sensitive", "all"], "benefit": "Cleanses without stripping moisture"},
            {"name": "Salicylic Acid Cleanser", "category": "cleanser", "for": ["acne_like", "pore_like"], "benefit": "Exfoliates inside pores"},
            {"name": "Hyaluronic Acid Serum", "category": "hydrator", "for": ["all", "dryness"], "benefit": "Deep hydration and plumping"},
            {"name": "Niacinamide Serum", "category": "brightener", "for": ["dark_spots", "pore_like"], "benefit": "Improves tone and texture"},
            {"name": "Vitamin C Serum", "category": "brightener", "for": ["dark_spots"], "benefit": "Brightens skin and protects"},
            {"name": "Ceramide Moisturizer", "category": "moisturizer", "for": ["sensitive", "all"], "benefit": "Repairs skin barrier"},
            {"name": "Non-comedogenic Moisturizer", "category": "moisturizer", "for": ["acne_like"], "benefit": "Hydrates without clogging pores"},
            {"name": "Broad Spectrum SPF 50", "category": "sunscreen", "for": ["all", "dark_spots"], "benefit": "Prevents further pigmentation"},
            {"name": "Centella Asiatica Cream", "category": "soothing", "for": ["redness", "sensitive"], "benefit": "Calms inflammation and redness"},
        ]

    def generate_routine(self, features):
        """
        Generates a complete AM/PM routine based on detected features.
        """
        am_routine = ["Gentle Cleanser", "Moisturizer", "Broad Spectrum SPF 50"]
        pm_routine = ["Gentle Cleanser", "Moisturizer"]

        # Add treatments based on findings
        if features.get('dark_spots', 0) > 2:
            am_routine.insert(1, "Vitamin C Serum")
            pm_routine.insert(1, "Niacinamide Serum")

        if features.get('acne_like_regions', 0) > 2:
            am_routine[0] = "Salicylic Acid Cleanser"
            pm_routine.insert(1, "Non-comedogenic Treatment")

        if features.get('redness_score', 0) > 0.1:
            pm_routine.insert(2, "Centella Asiatica Cream")

        return {
            "am": am_routine,
            "pm": pm_routine
        }

    def recommend(self, features):
        """
        Matches detected characteristics to product categories.
        """
        recommendations = []
        tags = []
        if features.get('redness_score', 0) > 0.1: tags.append("redness")
        if features.get('dark_spots', 0) > 2: tags.append("dark_spots")
        if features.get('acne_like_regions', 0) > 2: tags.append("acne_like")
        if features.get('pore_like_regions', 0) > 5: tags.append("pore_like")
        if features.get('skin_smoothness', 100) < 70: tags.append("texture")
        tags.append("all")

        for prod in self.product_db:
            if any(tag in prod['for'] for tag in tags):
                recommendations.append({
                    "name": prod['name'],
                    "category": prod['category'],
                    "benefit": prod['benefit']
                })

        return recommendations[:6]
