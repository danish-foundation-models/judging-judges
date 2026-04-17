"""
LLM Judge Evaluation for Danish Linguistic Quality

Usage:
    uv run python evaluate_llm_judge.py
"""

import random
import os
import re
from typing import Literal

import pandas as pd
from litellm import completion
from pydantic import BaseModel


class JudgeResponse(BaseModel):
    choice: Literal["A", "B"]
    reasoning: str


SYSTEM_PROMPT = """\
Du er en expert i danska sprog. Du får to setninger og skal afgøres hvilken der er best sprogligt.
Du skal svare med et JSON object som indeholder præcis 2 felter:
- "reasoning": en kort forklaring a dit valg (1-2 sætninger)
- "choice": enten "A" eller "B"

Eksempel på svar:
{
    "reasoning": "Sætning A er mere naturlig og grammatisk korrekt.",
    "choice": "A"
}

Svar kun med JSON-objektet. Inkludér ikke andet tekst eller markdown-formattering."""

def extract_json(content: str) -> str:
    # strip markdown code fences
    content = re.sub(r"```json|```", "", content)
    # extract the first {...} block, in case there's trailing text like ***
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {content!r}")
    return match.group()


def judge(model: str, sentence_a: str, sentence_b: str, api_base: str | None,
          api_key: str | None) -> JudgeResponse:
    use_structured_output = not model.startswith("together_ai")  # only use for native providers

    response = completion(
        api_base=api_base,
        api_key=api_key,
        drop_params=True,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Sætning A:\n{sentence_a}\n\nSætning B:\n{sentence_b}",
            },
        ],
        # temperature=0.0, # Set to 0 for deterministic output, but may cause issues with some models. Adjust as needed.
        response_format=JudgeResponse if use_structured_output else {"type": "json_object"},
    )
    content: str = response.choices[0].message.content  # type: ignore
    content = extract_json(content)
    try:
        return JudgeResponse.model_validate_json(content)
    except Exception as e:

        return JudgeResponse.model_validate_json(content)


def evaluate(df: pd.DataFrame, model: str, api_base: str | None, api_key: str | None,
             seed: int = 42,) -> pd.DataFrame:
    rng = random.Random(seed)
    records = []

    for _, row in df.iterrows():
        flip = rng.random() < 0.5
        a, b = (
            (row["bad sentence"], row["good sentence"])
            if flip
            else (row["good sentence"], row["bad sentence"])
        )
        correct = "B" if flip else "A"

        result = judge(model, a, b, api_base=api_base, api_key=api_key)
        records.append(
            {
                **row,
                "judge_model": model,
                "correct_choice": correct,
                "judge_choice": result.choice.strip().upper(),
                "judge_reasoning": result.reasoning,
                "is_correct": result.choice.strip().upper() == correct,
            }
        )
        print(result.choice.strip().upper() == correct)
    return pd.DataFrame(records)


if __name__ == "__main__":
    df = pd.read_csv("data/Linguistic_quality_preference_20260326.tsv", sep="\t")
    #df = df.head(10) # let us just test on 10 examples for now, but you can increase this as needed

    save_dir = "results/preference"
    os.makedirs(save_dir, exist_ok=True)

    custom_base_url = os.getenv("CUSTOM_BASE_URL")
    custom_api_key = os.getenv("CUSTOM_API_KEY")

    OLLAMA_BASE_URL = "http://localhost:11434"

    models: dict[str, str | None] = {
        # Large / high quality
        "ollama/llama3.3:70b": OLLAMA_BASE_URL,
        "ollama/qwen2.5:72b": OLLAMA_BASE_URL,
        "ollama/mistral-large": OLLAMA_BASE_URL,

        # Medium
        "ollama/llama3.1:8b": OLLAMA_BASE_URL,
        "ollama/qwen2.5:14b": OLLAMA_BASE_URL,
        "ollama/gemma3:12b": OLLAMA_BASE_URL,

        # Small / fast
        "ollama/llama3.2:3b": OLLAMA_BASE_URL,
        "ollama/qwen2.5:7b": OLLAMA_BASE_URL,
        "ollama/gemma3:4b": OLLAMA_BASE_URL,
    }

    for model_name, api_base in models.items():
        api_key=custom_api_key if api_base else None

        try:
            results = evaluate(
                df,
                model=model_name,
                api_base=api_base,
                api_key=api_key,
            )

        except Exception as e:
            print(f"Failed to evaluate {model_name}: {e}")
            continue

        safe_name = model_name.replace("/", "_")
        results.to_csv(f"{save_dir}/{safe_name}.tsv", sep="\t", index=False)

        print(f"Accuracy: {results['is_correct'].mean():.1%}")
        print(results.groupby("source")["is_correct"].mean())
        # split multi-value error types
        print(results
              .assign(error_type=results["error_types"].str.split("; "))
              .explode("error_type")
              .groupby("error_type")["is_correct"]
              .mean()
              )

