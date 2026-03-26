"""
LLM Judge Evaluation for Danish Linguistic Quality

Usage:
    uv run python evaluate_llm_judge.py
"""

import random
from typing import Literal

import pandas as pd
from litellm import completion
from pydantic import BaseModel


class JudgeResponse(BaseModel):
    choice: Literal["A", "B"]
    reasoning: str


SYSTEM_PROMPT = """\
Du er en ekspert i dansk sprog. Du får to sætninger og skal afgøre, hvilken der er bedst sprogligt."""


def judge(model: str, sentence_a: str, sentence_b: str) -> JudgeResponse:
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Sætning A:\n{sentence_a}\n\nSætning B:\n{sentence_b}",
            },
        ],
        # temperature=0.0, # Set to 0 for deterministic output, but may cause issues with some models. Adjust as needed.
        response_format=JudgeResponse,
    )
    content: str = response.choices[0].message.content  # type: ignore
    return JudgeResponse.model_validate_json(content)


def evaluate(df: pd.DataFrame, model: str, seed: int = 42) -> pd.DataFrame:
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

        result = judge(model, a, b)
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

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = pd.read_csv("data/Linguistic_quality_preference_20260326.tsv", sep="\t")
    df = df.head(10) # let us just test on 10 examples for now, but you can increase this as needed

    results = evaluate(
        df, model="gpt-4o"
    )  # we could make this into a loop to evaluate multiple models
    # for evaluting local models we can either use ollama or vllm to serve the model and then call it via the API,
    # e.g. using ollama - read more here https://docs.litellm.ai/
    results.to_csv("judge_results.csv", index=False)

    print(f"Accuracy: {results['is_correct'].mean():.1%}")
    print(results.groupby("source")["is_correct"].mean())
    print(results.groupby("error_types")["is_correct"].mean())
