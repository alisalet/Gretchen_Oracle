import os
import json
from enum import Enum
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


class TarotMode(str, Enum):
    LOVE = "love"
    CAREER = "career"


class TarotPredictionResponse(BaseModel):
    introduction: str = Field(description="Introduction describing the general atmosphere of the reading.")
    main_card_interpretation: str = Field(description="Analysis of the main card in the context of the chosen mode.")
    additional_cards_interpretation: str = Field(description="Analysis of the two additional cards.")
    synergy_and_warnings: str = Field(description="Interconnection of all three cards. Hidden nuances.")
    final_advice: str = Field(description="Final, wise, and supportive advice for the user's future.")


def generate_tarot_prediction(mode: str, main_card: str, additional_cards: List[str]) -> str:
    """Интегрированный ИИ-агент. Делает запрос к LLM и возвращает готовый текст."""
    system_prompt = (
        "You are a professional AI Tarot Reader. Your task is to interpret three Tarot cards "
        "based on the selected mode of operation (love or career). "
        "Adhere strictly to the chosen mode: if 'love', focus on relationships. "
        "If 'career', focus on work and finances. "
        "Respond ONLY with a raw JSON object matching the requested schema. "
        "Do not include any text outside the JSON object, no markdown wrappers like ```json."
    )

    user_prompt = f"""
    Reading Mode: {mode.lower()}
    Main Card: {main_card}
    Additional Cards: {', '.join(additional_cards)}

    Format your response EXACTLY as this JSON and nothing else. Ensure proper JSON syntax:
    {{
        "introduction": "string",
        "main_card_interpretation": "string",
        "additional_cards_interpretation": "string",
        "synergy_and_warnings": "string",
        "final_advice": "string"
    }}
    """

    print(f"Gretchen is thinking (Mode: {mode})...")

    try:
        response = client.chat.completions.create(
            model="google/gemma-4-26b-a4b-it:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            timeout=30
        )

        raw_content = response.choices[0].message.content.strip()
        cleaned_text = raw_content.replace("```json", "").replace("```", "").strip()
        json_dict = json.loads(cleaned_text)

        data = TarotPredictionResponse(**json_dict)

        full_story = (
            f"{data.introduction}\n\n"
            f"Speaking of the main force, {main_card}: {data.main_card_interpretation}\n\n"
            f"Regarding adjacent energies: {', '.join(additional_cards)}: {data.additional_cards_interpretation}\n\n"
            f"The hidden connection shows that {data.synergy_and_warnings}\n\n"
            f"My advice for you: {data.final_advice}"
        )
        return full_story

    except Exception as e:
        print(f"❌ AI Agent Error: {e}")
        return "The cosmic connection was interrupted. I cannot read the cards right now."