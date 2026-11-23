"""
Fertilizer Application Schedule Module
Generates agronomic split-application schedules based on crop growth stages,
planting date, and recommended fertilizer type.
"""

from datetime import datetime, timedelta
import pandas as pd
from src.database import add_schedule_task, get_all_schedules, update_schedule_task_status

# Crop cycle profiles (offsets in days relative to sowing/planting date)
CROP_CYCLE_PROFILES = {
    "Paddy": {
        "duration_days": 120,
        "stages": [
            ("Basal Application (At Transplanting)", 0, "Full dose of Phosphorus & Potassium; 1/3 dose of Nitrogen", "Soil incorporation before final puddling"),
            ("Tillering Stage Top Dressing", 21, "1/3 dose of Nitrogen (Urea)", "Broadcasting in 2-3 cm standing water"),
            ("Panicle Initiation Stage", 45, "Remaining 1/3 dose of Nitrogen + Foliar Micronutrients", "Broadcasting followed by light irrigation"),
            ("Heading / Grain Filling", 70, "Foliar spray of 00-00-50 (SOP) or dilute complex if needed", "Foliar spray during cool morning hours")
        ]
    },
    "Wheat": {
        "duration_days": 130,
        "stages": [
            ("Basal Application (Sowing Time)", 0, "Full Phosphorus & Potassium + 1/2 Nitrogen", "Drill application 5 cm below seed"),
            ("Crown Root Initiation (CRI)", 21, "1/4 dose of Nitrogen top dressing", "Broadcasting just before 1st irrigation"),
            ("Jointing / Tillering Stage", 45, "Remaining 1/4 dose of Nitrogen", "Top dressing followed by irrigation"),
            ("Flowering / Boot Stage", 75, "Foliar spray of Potassium Nitrate (13-0-45)", "Foliar spray for grain weight")
        ]
    },
    "Sugarcane": {
        "duration_days": 365,
        "stages": [
            ("Basal Application (Planting)", 0, "Full Phosphorus + 1/4 Nitrogen & Potassium", "Furrow application below setts"),
            ("Tillering Phase Top Dressing", 45, "1/4 dose Nitrogen & Potassium", "Side dressing followed by earthing up"),
            ("Early Grand Growth Stage", 90, "1/4 dose Nitrogen", "Side dressing followed by light irrigation"),
            ("Grand Growth Peak Stage", 120, "Final 1/4 dose Nitrogen & Potassium", "Side dressing and final earthing up")
        ]
    },
    "Cotton": {
        "duration_days": 160,
        "stages": [
            ("Basal Application", 0, "Full Phosphorus + 1/3 Nitrogen & Potassium", "Spot/banding application near seed holes"),
            ("Square Formation Stage", 35, "1/3 dose Nitrogen top dressing", "Side dressing 10 cm away from base"),
            ("Peak Flowering / Boll Formation", 65, "Remaining 1/3 dose Nitrogen + Potassium", "Side dressing followed by irrigation"),
            ("Boll Development", 90, "Foliar spray of Magnesium Sulphate & 13-0-45", "Foliar spray for boll retention")
        ]
    },
    "Maize": {
        "duration_days": 105,
        "stages": [
            ("Basal Application", 0, "Full Phosphorus & Potassium + 1/3 Nitrogen", "Band placement 5 cm beside seed line"),
            ("Knee-High Stage (V6)", 30, "1/3 dose Nitrogen top dressing", "Side dressing followed by intercultivation"),
            ("Tasseling / Silking Stage", 55, "Final 1/3 dose Nitrogen", "Broadcasting/side dressing before irrigation")
        ]
    },
    "Pulses": {
        "duration_days": 85,
        "stages": [
            ("Basal Application (Sowing)", 0, "Full recommended complex (DAP / 14-35-14 / 20-20)", "Drill application at seedbed preparation"),
            ("Early Branching Stage", 25, "Foliar spray of 2% DAP or Urea if nitrogen nodulation is slow", "Foliar spray in late afternoon"),
            ("Pod Formation Stage", 50, "Foliar spray of 1% Potassium Nitrate", "Foliar spray to prevent premature flower drop")
        ]
    },
    "Ground Nuts": {
        "duration_days": 110,
        "stages": [
            ("Basal Application", 0, "Full Phosphorus & Potassium + initial Nitrogen starter", "Incorporated into top 10 cm soil"),
            ("Gypsum Application (Pegging Stage)", 40, "Gypsum (Calcium & Sulphur source) 200-400 kg/ha", "Band application around pegging zone"),
            ("Pod Filling Stage", 65, "Foliar spray of micronutrient mixture & Potash", "Foliar spray for kernel plumpness")
        ]
    },
    "Millets": {
        "duration_days": 90,
        "stages": [
            ("Basal Application", 0, "Full Phosphorus & Potassium + 1/2 Nitrogen", "Drill application at sowing"),
            ("Vegetative Stage", 25, "Remaining 1/2 dose of Nitrogen", "Top dressing after weeding with soil moisture")
        ]
    },
    "Barley": {
        "duration_days": 115,
        "stages": [
            ("Basal Application", 0, "Full Phosphorus & Potassium + 1/2 Nitrogen", "Drilled along with seed furrow"),
            ("Active Tillering", 30, "Remaining 1/2 dose Nitrogen top dressing", "Top dressed prior to first irrigation")
        ]
    },
    "Tobacco": {
        "duration_days": 120,
        "stages": [
            ("Basal Transplanting Dose", 0, "Balanced complex (17-17-17 or 10-26-26)", "Furrow placement before transplanting"),
            ("Knee-High Stage", 35, "Top dressing with Potassium Nitrate (avoid chloride forms)", "Side dressing followed by light hoeing"),
            ("Topping Stage", 60, "Foliar application for leaf expansion", "Even foliar coverage")
        ]
    },
    "Oil seeds": {
        "duration_days": 100,
        "stages": [
            ("Basal Application", 0, "Full Phosphorus, Potassium & Sulphur + 1/2 Nitrogen", "Soil incorporation at last ploughing"),
            ("Flowering Stage", 35, "Remaining 1/2 Nitrogen top dressing", "Top dressing followed by irrigation"),
            ("Siliqua / Capsule Development", 60, "Foliar spray of 1% Potassium Nitrate", "Foliar spray to boost oil synthesis")
        ]
    }
}

# Generic fallback profile
DEFAULT_PROFILE = {
    "duration_days": 100,
    "stages": [
        ("Basal Application", 0, "100% Phosphorus, 100% Potassium, 50% Nitrogen", "Soil incorporation during seedbed preparation"),
        ("Vegetative Stage Top Dressing", 30, "Remaining 50% Nitrogen", "Side dressing followed by light irrigation"),
        ("Reproductive / Flowering Stage", 60, "Foliar maintenance spray if needed", "Foliar spray during non-peak sun hours")
    ]
}


def generate_schedule(crop_name: str, planting_date: datetime.date, fertilizer_name: str, farm_name: str = "My Farm"):
    """
    Generates tailored application schedule dates and descriptions based on crop profile.
    """
    profile = CROP_CYCLE_PROFILES.get(crop_name, DEFAULT_PROFILE)
    schedule_tasks = []

    for stage_name, day_offset, dose_desc, method in profile["stages"]:
        target_dt = planting_date + timedelta(days=day_offset)
        target_str = target_dt.strftime("%Y-%m-%d")

        # Refine dose description with specific recommended fertilizer
        refined_dose = f"{fertilizer_name} complex: {dose_desc}"

        task = {
            "farm_name": farm_name,
            "crop_name": crop_name,
            "fertilizer_name": fertilizer_name,
            "planting_date": planting_date.strftime("%Y-%m-%d"),
            "stage_name": stage_name,
            "target_date": target_str,
            "recommended_dose_desc": refined_dose,
            "application_method": method,
            "status": "Pending",
            "notes": f"Scheduled for {stage_name} on {target_str}."
        }
        schedule_tasks.append(task)

    return schedule_tasks


def save_schedule_to_db(schedule_tasks: list):
    """
    Saves a generated schedule list into the SQLite database.
    """
    task_ids = []
    for t in schedule_tasks:
        tid = add_schedule_task(
            farm_name=t["farm_name"],
            crop_name=t["crop_name"],
            fertilizer_name=t["fertilizer_name"],
            planting_date=t["planting_date"],
            stage_name=t["stage_name"],
            target_date=t["target_date"],
            recommended_dose_desc=t["recommended_dose_desc"],
            application_method=t["application_method"],
            status=t["status"],
            notes=t["notes"]
        )
        task_ids.append(tid)
    return task_ids


def get_safety_precautions():
    """
    Returns essential agronomic safety precautions.
    """
    return [
        "**Protective Equipment**: Wear waterproof gloves, protective eyewear, and a dust mask when handling chemical fertilizers.",
        "**Weather Consideration**: Avoid broadcasting fertilizers immediately prior to heavy rainfall to minimize nutrient runoff into local water bodies.",
        "**Soil Moisture**: Ensure soil is moist before applying nitrogenous fertilizers like Urea to avoid root scorch and ammonia volatilization.",
        "**Safe Storage**: Store fertilizer bags in a dry, well-ventilated, shaded facility elevated on wooden pallets, away from cattle, pets, and children.",
        "**Scientific Verification**: Fertilizer application schedules provided are general agronomic guidance. Field rates must be adjusted based on certified laboratory soil test reports (STCR) and local agricultural extension recommendations."
    ]
