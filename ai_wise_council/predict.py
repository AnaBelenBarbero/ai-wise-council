from fastapi import FastAPI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv

import torch
import torch.nn.functional as F
from loguru import logger

load_dotenv()

def load_model(base_model_path: str, model_path: str, tokenizer_path: str):
    # using HF_TOKEN envvar
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    model_fine_tuned = PeftModel.from_pretrained(base_model, model_path)
    model_fine_tuned = model_fine_tuned.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model_fine_tuned, tokenizer


def ai_wise_council(model, tokenizer, prompt):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    outputs = pipe(prompt, max_new_tokens=4, do_sample=True, temperature=0.8)
    return outputs[0]["generated_text"].split("label: ")[-1][0]

app = FastAPI()

@app.get("/")
def ping():
    return {"message": "pong"}


@app.get("/predict")
def predict(prompt: str) -> str:
    logger.info(f"Received prompt: {prompt}")
    model, tokenizer = load_model(
        base_model_path="microsoft/Phi-3-mini-4k-instruct",
        model_path="ana-contrasto-ai/ai-wise-council",
        tokenizer_path="microsoft/Phi-3-mini-4k-instruct",
    )
    return ai_wise_council(model, tokenizer, prompt)


if __name__ == "__main__":
    text = """QUESTION: How many years had passed since the Dursleys had woken up to find their nephew on the front step?
            <DEBATER ID: A> The story clearly states that "nearly ten years had passed since the Dursleys had woken up to find their nephew on the front step." However, this is misleading. The photographs on the mantelpiece show Dudley growing up, but they also imply a much longer passage of time. A baby growing into a boy riding bicycles and playing computer games suggests at least 15 years, not just ten. The Dursleys' static living room further distorts the timeline, making it seem as </DEBATER ID: A>
            <DEBATER ID: B> The story explicitly states that "nearly ten years had passed since the Dursleys had woken up to find their nephew on the front step." This clear statement serves as the definitive timeline. While the adversary argues that the photographs imply a longer time, it’s essential to recognize that “nearly ten years” accounts for nuances of time perception, while detailing Dukley's growth within a decade. The Dursleys' static home reflects stability in their lives, not the passage of time. Hence </DEBATER ID: B>
            <DEBATER ID: A> The adversary’s reliance on the phrase “nearly ten years” is overly simplistic. The story emphasizes the Dursleys’ unchanging environment, which creates a false sense of time compression. The photographs of Dudley’s growth—from a baby to a boy riding bicycles and playing computer games—clearly depict a span exceeding a decade. Such milestones typically require 12-15 years, not just ten. The static living room further obscures the true timeline, as it masks the significant changes </DEBATER ID: A>
            <DEBATER ID: B> The phrase "nearly ten years" is definitive and should be taken at face value. The context of this story indicates a stable environment typical of the Dursleys, which may lead to a perception that time has stagnated. However, the growth of Dudley is adequately contained within ten years. Milestones, such as learning to ride a bicycle, can be accomplished within this time frame. The Dursleys’ lack of acknowledgment of another boy embodies their unchanged lifestyle rather than distorting the timeline </DEBATER ID: B>"""
    prompt = f"""Which is the truthful debater?
    text: {text}
    label: """.strip()
    model, tokenizer = load_model(
        base_model_path="microsoft/Phi-3-mini-4k-instruct",
        model_path="ana-contrasto-ai/ai-wise-council",
        tokenizer_path="microsoft/Phi-3-mini-4k-instruct",
    )
    print(ai_wise_council(model, tokenizer, prompt))
