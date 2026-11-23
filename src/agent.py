"""
AI Agricultural Assistant Module
An intelligent, offline, rule-and-knowledge-based agricultural reasoning agent.
Features:
- Answers queries on fertilizers, soil health, crop nutrient requirements, and safety.
- Explains machine learning recommendations and N-P-K nutrient dynamics.
- Operates entirely offline without external paid LLM API keys.
- Completely transparent: Never claims to be an external LLM.
"""

import re
from datetime import datetime

# Domain Knowledge Base
FERTILIZER_KNOWLEDGE = {
    "urea": {
        "name": "Urea (46-0-0)",
        "npk": "46% Nitrogen, 0% Phosphorus, 0% Potassium",
        "role": "Highest nitrogen concentration among solid fertilizers. Essential for vegetative growth, deep green foliage, protein synthesis, and chlorophyll formation.",
        "usage": "Split applications recommended. Apply as basal or top dressing during active vegetative stages. Never apply on dry soil or leave uncovered in hot sun to prevent ammonia volatilization.",
        "crops": "Paddy, Wheat, Sugarcane, Maize, Cotton."
    },
    "dap": {
        "name": "DAP (Diammonium Phosphate 18-46-0)",
        "npk": "18% Nitrogen, 46% Phosphorus (P2O5), 0% Potassium",
        "role": "High source of water-soluble phosphate with starter nitrogen. Critical for early root elongation, strong seedling anchorage, and tillering.",
        "usage": "Best applied at planting (basal application) placed 3-5 cm below and to the side of seeds.",
        "crops": "Wheat, Pulses, Oil seeds, Paddy, Ground Nuts."
    },
    "14-35-14": {
        "name": "14-35-14 (Complex Fertilizer)",
        "npk": "14% Nitrogen, 35% Phosphorus, 14% Potassium",
        "role": "High-phosphorus complex with balanced N and K. Stimulates robust root systems, early floral differentiation, and balanced plant vigor.",
        "usage": "Ideal basal dressing for phosphorus-deficient soils and pulse or oilseed crops.",
        "crops": "Ground Nuts, Pulses, Cotton, Oil seeds."
    },
    "28-28": {
        "name": "28-28-0 (Nitrophosphate Complex)",
        "npk": "28% Nitrogen, 28% Phosphorus, 0% Potassium",
        "role": "1:1 ratio of nitrogen and phosphorus. Maximizes simultaneous canopy development and root expansion.",
        "usage": "Excellent starter for crops grown in soils with adequate native potassium.",
        "crops": "Sugarcane, Paddy, Maize, Cotton."
    },
    "20-20": {
        "name": "20-20-0 (Ammonium Phosphate Sulphate)",
        "npk": "20% Nitrogen, 20% Phosphorus + 13% Sulphur",
        "role": "Balanced NP with crucial secondary sulphur. Enhances oil content in oilseeds and protein in legumes.",
        "usage": "Broadcasting or side dressing during early growth stages.",
        "crops": "Oil seeds, Pulses, Cotton, Ground Nuts."
    },
    "17-17-17": {
        "name": "17-17-17 (Complete Balanced Complex)",
        "npk": "17% Nitrogen, 17% Phosphorus, 17% Potassium",
        "role": "Equal ratio of all three macronutrients. Ensures balanced vegetative, root, and reproductive development.",
        "usage": "Basal or top dressing for continuous feeder crops.",
        "crops": "Sugarcane, Tobacco, Vegetables, Maize, Millets."
    },
    "10-26-26": {
        "name": "10-26-26 (High Potash & Phosphate Complex)",
        "npk": "10% Nitrogen, 26% Phosphorus, 26% Potassium",
        "role": "High potassium and phosphorus content. Improves water regulation, drought tolerance, stalk strength, grain filling, and disease immunity.",
        "usage": "Basal application and top dressing during flowering/heading stages.",
        "crops": "Sugarcane, Paddy, Cotton, Tobacco."
    }
}

SOIL_KNOWLEDGE = {
    "clayey": {
        "type": "Clayey Soil",
        "properties": "Fine texture, high nutrient and water retention, slow drainage, prone to waterlogging and compaction.",
        "recommendation": "Incorporate organic matter or green manure to improve aeration. Split fertilizer doses to avoid salt accumulation. Avoid excessive heavy irrigation."
    },
    "sandy": {
        "type": "Sandy Soil",
        "properties": "Coarse texture, rapid drainage, low water holding capacity, prone to rapid nutrient leaching.",
        "recommendation": "Apply fertilizers in multiple small, frequent split doses (fertigation or top dressings). Apply compost/vermicompost to boost cation exchange capacity."
    },
    "loamy": {
        "type": "Loamy Soil",
        "properties": "Ideal agricultural soil with balanced sand, silt, and clay. Excellent moisture retention, good drainage, and high nutrient availability.",
        "recommendation": "Maintain organic matter through crop rotation. Standard recommended fertilizer split schedules work optimally."
    },
    "black": {
        "type": "Black (Regur) Soil",
        "properties": "High clay content, rich in calcium, potassium, and magnesium, but generally deficient in nitrogen and phosphorus. Highly moisture-retentive, cracks when dry.",
        "recommendation": "Deep furrow application for basal fertilizers. Supplemental nitrogen and phosphorus applications yield significant response. Excellent for Cotton and Sugarcane."
    },
    "red": {
        "type": "Red Soil",
        "properties": "Porous, friable structure, rich in iron and potash, but deficient in nitrogen, phosphorus, and organic humus. Slightly acidic to neutral.",
        "recommendation": "Requires regular additions of organic farmyard manure. High response to phosphate fertilizers (DAP or 14-35-14) and nitrogen."
    }
}

NUTRIENT_KNOWLEDGE = {
    "nitrogen": {
        "symbol": "N",
        "function": "Vegetative shoot and leaf growth, chlorophyll formation, and protein synthesis.",
        "deficiency": "Stunted growth, pale yellowish leaves starting from older bottom leaves (chlorosis).",
        "excess": "Excessive leafy growth, dark green weak stems, delayed flowering, lodging risk, and susceptibility to pests."
    },
    "phosphorous": {
        "symbol": "P",
        "function": "Root development, seed germination, flowering, fruit setting, and cellular energy transfer (ATP).",
        "deficiency": "Purplish or reddish discoloration on leaves and stems, stunted root growth, delayed maturity.",
        "excess": "Interferes with zinc and iron uptake, leading to micronutrient deficiencies."
    },
    "potassium": {
        "symbol": "K",
        "function": "Stomatal regulation, water stress tolerance, enzyme activation, stalk strength, grain plumpness, and pest resistance.",
        "deficiency": "Marginal leaf scorching, browning of leaf tips (marginal chlorosis), weak lodging-prone stems.",
        "excess": "Can impair calcium and magnesium uptake."
    }
}


class AgriculturalAssistant:
    """
    Offline Agricultural AI Assistant that matches user queries using domain NLP heuristics.
    """

    def __init__(self):
        self.conversation_history = []

    def get_welcome_message(self):
        return (
            "Hello! I am your **Intelligent Agricultural Assistant**. "
            "I provide evidence-based guidance on fertilizers, soil health, N-P-K nutrient balancing, "
            "crop requirements, and application safety.\n\n"
            "💡 *You can ask me questions like:*\n"
            "- *What is DAP and when should I use it?*\n"
            "- *How do I manage fertilizer for Sandy soil?*\n"
            "- *What are the symptoms of Nitrogen deficiency?*\n"
            "- *Which fertilizer is best for Sugarcane or Cotton?*\n"
            "- *What precautions should I take when applying Urea?*\n"
            "- *Explain the 10-26-26 fertilizer.*"
        )

    def process_query(self, query: str, context: dict = None) -> str:
        """
        Processes a farmer/student query and returns a detailed agronomic response.
        """
        q = query.lower().strip()

        # Check for empty query
        if not q:
            return "Please enter a question about fertilizers, soil, crops, or nutrient management."

        # 1. Greetings
        if re.search(r"\b(hi|hello|hey|greetings|namaste)\b", q):
            return (
                "Hello! Welcome to the Agricultural Assistant. How can I help you today with your soil, crops, or fertilizer planning?"
            )

        # 2. Inquire about LLM identity
        if re.search(r"\b(what are you|who are you|are you an llm|chatgpt|openai|gemini)\b", q):
            return (
                "I am a specialized **Agricultural Expert Agent** built specifically for this Fertilizer Recommendation project. "
                "I operate offline using an integrated agronomic knowledge engine and rule-based agricultural reasoning system. "
                "I do not use external cloud LLM APIs, ensuring zero API costs, high reliability, and privacy."
            )

        # 3. Contextual question about current session recommendation
        if context and re.search(r"\b(explain my recommendation|why this fertilizer|my prediction|recommendation)\b", q):
            fert = context.get("recommended_fertilizer")
            if fert:
                crop = context.get("crop_type", "your crop")
                soil = context.get("soil_type", "your soil")
                return (
                    f"**Analysis of Your Recommendation ({fert}):**\n\n"
                    f"The machine learning model recommended **{fert}** based on the specific nutrient profile you submitted for {crop} in {soil} soil.\n\n"
                    f"- **Nitrogen status:** {context.get('nitrogen', 'N/A')}\n"
                    f"- **Phosphorous status:** {context.get('phosphorous', 'N/A')}\n"
                    f"- **Potassium status:** {context.get('potassium', 'N/A')}\n\n"
                    f"This fertilizer provides the precise balance needed to correct nutrient limitations while avoiding excessive salt accumulation."
                )

        # 4. Specific Fertilizer Queries
        for key, data in FERTILIZER_KNOWLEDGE.items():
            if re.search(r"\b" + re.escape(key) + r"\b", q):
                return (
                    f"### 🌾 {data['name']}\n\n"
                    f"- **NPK Ratio:** {data['npk']}\n"
                    f"- **Agronomic Role:** {data['role']}\n"
                    f"- **Application Method:** {data['usage']}\n"
                    f"- **Common Crops:** {data['crops']}\n\n"
                    f"⚠️ *Note: Application rates should always be verified with local soil test results.*"
                )

        # 5. Soil Types Queries
        for key, data in SOIL_KNOWLEDGE.items():
            if re.search(r"\b" + re.escape(key) + r"\b", q):
                return (
                    f"### 🌱 {data['type']} Management\n\n"
                    f"- **Soil Properties:** {data['properties']}\n"
                    f"- **Fertilizer & Water Advice:** {data['recommendation']}\n\n"
                    f"💡 *Tip: Regular application of well-decomposed organic farmyard manure (FYM) enhances nutrient uptake efficiency across all soil classes.*"
                )

        # 6. Primary Macronutrients (N, P, K)
        for key, data in NUTRIENT_KNOWLEDGE.items():
            if re.search(r"\b" + re.escape(key) + r"\b", q):
                return (
                    f"### 🧪 {key.capitalize()} ({data['symbol']}) Agronomic Profile\n\n"
                    f"- **Core Plant Function:** {data['function']}\n"
                    f"- **Deficiency Symptoms:** {data['deficiency']}\n"
                    f"- **Excess Symptoms:** {data['excess']}\n\n"
                    f"💡 *Tip: Conducting a periodic Soil Test (STCR method) is the most reliable way to monitor {key.capitalize()} levels.*"
                )

        # 7. Crop-Specific Fertilizer Needs
        crop_keywords = {
            "paddy": ("Paddy (Rice)", "High nitrogen and balanced phosphorus demand. Requires 3 split doses: Basal (P, K, 1/3 N), Active Tillering (1/3 N), and Panicle Initiation (1/3 N)."),
            "wheat": ("Wheat", "Crown Root Initiation (CRI) at 21 days is the most critical moisture and nitrogen stage. Balanced DAP/NPK at sowing followed by Urea top dressing."),
            "sugarcane": ("Sugarcane", "Heavy nutrient feeder. High nitrogen requirement during tillering and grand growth, with high potassium (10-26-26) to maximize cane weight and sugar recovery."),
            "cotton": ("Cotton", "Requires careful nitrogen management to prevent excessive vegetative growth over boll formation. Phosphorus and Potassium (DAP, 10-26-26) ensure boll retention."),
            "maize": ("Maize", "Rapid vegetative consumer. Knee-high stage and tasseling require timely nitrogen top dressings."),
            "pulses": ("Pulses / Legumes", "Fixes atmospheric nitrogen through root nodules. Primary fertilizer need is Phosphorus (DAP or 14-35-14) to support root nodulation."),
            "ground nuts": ("Ground Nuts", "Requires Calcium and Sulphur (Gypsum) at pegging stage (~40 days) along with basal phosphorus for well-filled pods.")
        }

        for ckey, (cname, cdesc) in crop_keywords.items():
            if re.search(r"\b" + re.escape(ckey) + r"\b", q):
                return (
                    f"### 🌾 Nutrient Management for {cname}\n\n"
                    f"{cdesc}\n\n"
                    f"💡 *Check the **Fertilizer Application Schedule** page to generate a timeline for {cname}.*"
                )

        # 8. Safety & Precautions
        if re.search(r"\b(safety|precautions|protective|handling|store|storage|danger|poison)\b", q):
            return (
                "### 🛡️ Fertilizer Handling & Environmental Safety Guidelines\n\n"
                "1. **Personal Protective Equipment (PPE):** Always wear gloves, dust masks, and safety goggles when handling and broadcasting chemical fertilizers.\n"
                "2. **Prevent Water Pollution:** Maintain a 10-meter buffer strip from canals, ponds, and borewells. Never apply before anticipated downpours.\n"
                "3. **Soil Moisture Control:** Do not broadcast Urea on bone-dry soil; apply when the soil has adequate moisture or irrigate promptly.\n"
                "4. **Proper Storage:** Store fertilizer bags off the ground on wooden pallets in a cool, dry, ventilated store room away from livestock feeds and children.\n"
                "5. **Disposal:** Dispose of empty bags responsibly according to local agricultural waste regulations."
            )

        # 9. Application Methods
        if re.search(r"\b(method|apply|broadcasting|fertigation|foliar|banding|side dressing)\b", q):
            return (
                "### 🚜 Fertilizer Application Methods\n\n"
                "- **Basal Placement:** Placing fertilizer in the furrow below or beside seeds at sowing. Best for Phosphorus and Potassium which are relatively immobile in soil.\n"
                "- **Top Dressing / Broadcasting:** Spreading fertilizer uniformly across the crop field. Primarily used for soluble Nitrogen (Urea) during active growth.\n"
                "- **Side Dressing:** Applying fertilizer alongside crop rows 5-10 cm away from plant stems, then incorporating lightly into the soil.\n"
                "- **Foliar Spray:** Spraying dilute water-soluble nutrients directly onto leaf canopies. Ideal for rapid correction of micronutrients or during drought stress.\n"
                "- **Fertigation:** Injecting water-soluble fertilizers directly through drip or sprinkler irrigation systems."
            )

        # Fallback response
        return (
            "I understand you are asking about agricultural nutrient management. "
            "To give you the most accurate answer, please specify whether you want details on:\n\n"
            "- A specific fertilizer: *Urea, DAP, 14-35-14, 28-28, 20-20, 17-17-17, 10-26-26*\n"
            "- A specific soil type: *Clayey, Sandy, Loamy, Black, Red*\n"
            "- A crop: *Paddy, Wheat, Sugarcane, Cotton, Maize, Pulses, Ground Nuts*\n"
            "- A macronutrient: *Nitrogen (N), Phosphorous (P), Potassium (K)*\n"
            "- Fertilizer safety or application methods."
        )


# Global assistant instance
assistant = FertilizerPredictorAssistant = AgriculturalAssistant()
