"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.18.9
"""
import wandb
from transformers import (
    pipeline as transformer_pipeline,
    set_seed
)

def few_shot_yes_no(question):
    prompt = f"""Pytanie: "Czy Słońce jest jasne?"
                Odpowiedź: tak
                Pytanie: "Czy Ziemia jest płaska?"
                Odpowiedź: nie
                Pytanie: "Czy człowiek jest ssakiem?"
                Odpowiedź: tak
                Pytanie: "Czy banany rosną na Marsie?"
                Odpowiedź: nie
                Pytanie: "{question}"
                Odpowiedź:"""
    return prompt



def one_shot_yes_no(question):
    prompt = f"""Pytanie: "{question}"
    Odpowiedź:"""
    return prompt


def test_performance_fs(model, tokenizer, questions, answers, trained = False, seed=42):
    generator = transformer_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
    )
    set_seed(seed)
    accuracy = 0
    positive_accuracy = 0
    negative_accuracy = 0
    negs = 0
    pos = 0

    for question, answer in zip(questions, answers):
        fs_question = few_shot_yes_no(question)
        tokenized_question = tokenizer.encode(fs_question, return_tensors="pt")

        model_output = generator(
            fs_question,
            # end_sequence='###',
            return_full_text=False,
            do_sample=True,
            max_new_tokens=10,
            #top_k=5,
        )

        model_text = model_output[0]["generated_text"]
        first_token_lower = model_text.split(" ")[1].lower()

        print(f"Question: {question}")
        print(f"Model answer: {first_token_lower}")
        print(f"Real answer: {answer}")
        if answer == 'tak':
            pos += 1
            if answer == first_token_lower[:3]:
                accuracy += 1
                positive_accuracy += 1
        
        if answer == 'nie':
            negs += 1
            if answer == first_token_lower[:3]:
                accuracy += 1
                negative_accuracy += 1


    if wandb.run is not None:
        if trained == False:
            wandb.log({"Pre-tuning FS Accuracy" : accuracy / len(questions)})
            wandb.log({"Pre-tuning FS PositiveAccuracy" : positive_accuracy / pos})
            wandb.log({"Pre-tuning FS NegativeAccuracy" : negative_accuracy / negs})
        else:
            wandb.log({"After-tuning FS Accuracy" : accuracy / len(questions)})
            wandb.log({"After-tuning FS PositiveAccuracy" : positive_accuracy / pos})
            wandb.log({"After-tuning FS NegativeAccuracy" : negative_accuracy / negs})
    return accuracy / len(questions)

def test_performance_os(model, tokenizer, questions, answers, trained = False, seed=42):
    generator = transformer_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
    )
    set_seed(seed)
    
    accuracy = 0
    positive_accuracy = 0
    negative_accuracy = 0
    negs = 0
    pos = 0

    for question, answer in zip(questions, answers):
        fs_question = one_shot_yes_no(question)
        tokenized_question = tokenizer.encode(fs_question, return_tensors="pt")

        model_output = generator(
            fs_question,
            # end_sequence='###',
            return_full_text=False,
            do_sample=True,
            max_new_tokens=10,
        )

        model_text = model_output[0]["generated_text"]
        first_token_lower = model_text.split(" ")[1].lower()

        print(f"Question: {question}")
        print(f"Model answer: {first_token_lower}")
        print(f"Real answer: {answer}")
        
        if answer == 'tak':
            pos += 1
            if answer == first_token_lower[:3]:
                accuracy += 1
                positive_accuracy += 1
        
        if answer == 'nie':
            negs += 1
            if answer == first_token_lower[:3]:
                accuracy += 1
                negative_accuracy += 1

    print(accuracy / len(questions))
    if wandb.run is not None:
        if trained == False:
            wandb.log({"Pre-tuning OS Accuracy" : accuracy / len(questions)})
            wandb.log({"Pre-tuning OS PositiveAccuracy" : positive_accuracy / pos})
            wandb.log({"Pre-tuning OS NegativeAccuracy" : negative_accuracy / negs})
        else:
            wandb.log({"After-tuning OS Accuracy" : accuracy / len(questions)})
            wandb.log({"After-tuning OS PositiveAccuracy" : positive_accuracy / pos})
            wandb.log({"After-tuning OS NegativeAccuracy" : negative_accuracy / negs})
    return accuracy / len(questions)