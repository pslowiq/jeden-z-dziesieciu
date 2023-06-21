"""
This is a boilerplate pipeline 'binary_questions'
generated using Kedro 0.18.9
"""
import wandb
from transformers import (
    AutoModelForCausalLM,
    pipeline as transformer_pipeline,
    set_seed,
)

from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, TrainerCallback

import os


def load_model(model_name):
    if model_name == "papugapt2":
        model = AutoModelForCausalLM.from_pretrained("flax-community/papuGaPT2")
    if model_name == "gpt2":
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    if model_name == "gpt4all":
        model = AutoModelForCausalLM.from_pretrained("nomic-ai/gpt4all-j", revision="v1.2-jazzy")
    return model

def train(model_params, model, tokenizer, train_filepath, test_questions, test_answers):

    training_params = model_params['training_params']
    os.environ['WANDB_PROJECT'] = "Jeden z Dziesięciu"
    wandb.init()
    test_performance_fs_samples_perm(model, tokenizer, test_questions, test_answers, 0)
    test_performance_fs(model, tokenizer, test_questions, test_answers, 0)
    test_performance_fs_samples(model, tokenizer, test_questions, test_answers, 0)

    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_filepath,
        block_size=training_params['block_size'],
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.unk_token = ' '
    tokenizer.max_length = 1024
    tokenizer.add_special_tokens = False


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    class EvalCallback(TrainerCallback):
        def __init__(self, funs, model):
            super().__init__()
            self.funs = funs
            self.model = model

        def on_epoch_end(self, args, state, control, logs=None, **kwargs):
            for fun in self.funs:
                fun(self.model, tokenizer, test_questions, test_answers, state.epoch)


    training_args = TrainingArguments(
        output_dir = 'logs/',
        per_device_train_batch_size=training_params['batch_size'],
        num_train_epochs=training_params['epochs'],
        lr_scheduler_type=training_params['lr_scheduler_type'],
        learning_rate=training_params['learning_rate'],
        weight_decay=training_params['weight_decay'],
        warmup_steps=training_params['warmup_steps'],
        gradient_accumulation_steps=training_params['gradient_accumulation_steps'],
        report_to='wandb',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        callbacks = [
            EvalCallback([test_performance_fs, test_performance_fs_samples, test_performance_fs_samples_perm], model),
        ]
    )

    trainer.train()
    trainer.save_model()

    model.cpu()
    return model

Q1, A1 = "Czy Słońce jest jasne?", "tak"
Q2, A2 = "Czy Ziemia jest płaska?", "nie"
Q3, A3 = "Czy człowiek jest ssakiem?", "tak"
Q4, A4 = "Czy banany rosną na Marsie?", "nie"
DEFAULT_SHOT = ((Q1, A1), (Q2, A2), (Q3, A3), (Q4, A4)) 

def few_shot_yes_no(question, shot = DEFAULT_SHOT):
    prompt = ''
    for tup in shot:
        q, a = tup
        prompt += f"Pytanie: {q}\nOdpowiedź: {a}\n###\n"

    prompt += f"Pytanie: {question}\nOdpowiedź:"
    return prompt



def one_shot_yes_no(question):
    prompt = f"""Pytanie: "{question}"
    Odpowiedź:"""
    return prompt


def test_performance_fs(model, tokenizer, questions, answers, epoch, seed=42):
    model.cpu()
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
        #tokenized_question = tokenizer.encode(fs_question, return_tensors="pt")

        model_output = generator(
            fs_question,
            return_full_text=False,
            do_sample=True,
            max_new_tokens=10,
        )

        model_text = model_output[0]["generated_text"]
        first_token_lower = model_text.split(" ")[1].lower()

        #print(f"Question: {question}")
        #print(f"Model answer: {first_token_lower}")
        #print(f"Real answer: {answer}")
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
            elif 'tak' != first_token_lower[:3]:
                negs -= 1


    if wandb.run is not None:
        wandb.log({f"FS Accuracy" : accuracy / len(questions),
                   f"FS PositiveAccuracy" : positive_accuracy / pos,
                   f"FS NegativeAccuracy" : negative_accuracy / negs,
                   f"Epoch" : epoch})
    
    model.cuda()


def test_performance_fs_samples(model, tokenizer, questions, answers, epoch, samples = 5, seed=42):
    model.cpu()
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
        #tokenized_question = tokenizer.encode(fs_question, return_tensors="pt")

        model_output = generator(
            fs_question,
            return_full_text=False,
            do_sample=True,
            max_new_tokens=10,
            num_return_sequences=samples
        )

        tak_cnt = 0
        nie_cnt = 0
        for i in range(min(samples, len(model_output))):

            model_text = model_output[i]["generated_text"]
            first_token_lower = model_text.split(" ")[1].lower()

            if first_token_lower[:3] == 'tak':
                tak_cnt += 1
            if first_token_lower[:3] == 'nie':
                nie_cnt += 1


        #print(f"Question: {question}")
        #print(f"Model answer: {first_token_lower}")
        #print(f"Real answer: {answer}")

        model_answer = 'tak' if tak_cnt>nie_cnt else 'nie'
        if answer == 'tak':
            pos += 1
            if answer == model_answer:
                accuracy += 1
                positive_accuracy += 1
        
        if answer == 'nie':
            negs += 1
            if answer == model_answer:
                accuracy += 1
                negative_accuracy += 1
            elif 'tak' != model_answer:
                negs -= 1


    if wandb.run is not None:
        wandb.log({f"FS-{samples}-SamplesMV Accuracy" : accuracy / len(questions),
                   f"FS-{samples}-SamplesMV PositiveAccuracy" : positive_accuracy / pos,
                   f"FS-{samples}-SamplesMV NegativeAccuracy" : negative_accuracy / negs,
                   f"Epoch" : epoch})
    
    model.cuda()

from itertools import permutations
from random import sample

def test_performance_fs_samples_perm(model, tokenizer, questions, answers, epoch, samples = 5, n_perms = 5, seed=42):
    model.cpu()
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

    perms = list(permutations(DEFAULT_SHOT))

    for question, answer in zip(questions, answers):
            
        perms = sample(perms, k = n_perms)
        perms_tak = 0
        perms_nie = 0
        for perm in perms:
            fs_question = few_shot_yes_no(question, perm)
            #tokenized_question = tokenizer.encode(fs_question, return_tensors="pt")
            tak_cnt = 0
            nie_cnt = 0

            model_output = generator(
                fs_question,
                return_full_text=False,
                do_sample=True,
                max_new_tokens=10,
                num_return_sequences=samples
            )
            if samples != len(model_output):
                print(len(model_output))

            for i in range(min(samples, len(model_output))):

                model_text = model_output[i]["generated_text"]
                if len(model_text.split(" ")) < 2:
                    continue

                first_token_lower = model_text.split(" ")[1].lower()

                if first_token_lower[:3] == 'tak':
                    tak_cnt += 1
                if first_token_lower[:3] == 'nie':
                    nie_cnt += 1
            if tak_cnt>nie_cnt:
                perms_tak += 1
            else:
                perms_nie += 1

        #print(f"Question: {question}")
        #print(f"Model answer: {first_token_lower}")
        #print(f"Real answer: {answer}")

        model_answer = 'tak' if perms_tak>perms_nie else 'nie'
        if answer == 'tak':
            pos += 1
            if answer == model_answer:
                accuracy += 1
                positive_accuracy += 1
        
        if answer == 'nie':
            negs += 1
            if answer == model_answer:
                accuracy += 1
                negative_accuracy += 1
            elif 'tak' != model_answer:
                negs -= 1


    if wandb.run is not None:
        wandb.log({f"FS-{samples}-Samples-{n_perms}-PermsMV Accuracy" : accuracy / len(questions),
                   f"FS-{samples}-Samples-{n_perms}-PermsMV PositiveAccuracy" : positive_accuracy / pos,
                   f"FS-{samples}-Samples-{n_perms}-PermsMV NegativeAccuracy" : negative_accuracy / negs,
                   f"Epoch" : epoch})
    
    model.cuda()

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