import datasets
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, HfArgumentParser
import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
import numpy as np

NUM_PREPROCESSING_WORKERS = 2

def load_data_set(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def prepare_dataset_nli(examples, tokenizer, max_length):
    # Tokenize the premises and hypotheses
    tokenized_inputs = tokenizer(
        examples["premise"], 
        examples["hypothesis"], 
        max_length=max_length, 
        truncation=True, 
        padding="max_length",
        return_tensors="pt"
    )

    # Convert labels to tensors
    if "label" in examples:
        tokenized_inputs["labels"] = torch.tensor(examples["label"])

    return tokenized_inputs

def evaluate_and_print_individual_examples(trainer, tokenizer, data, args, set_name):
    print(f"\nIndividual Predictions for {set_name} Set:")
    for example in data:
        # Convert the example to the format expected by the model
        processed_example = prepare_dataset_nli({"premise": [example["premise"]], "hypothesis": [example["hypothesis"]], "label": [example["label"]]}, tokenizer, args.max_length)
        
        # Convert processed example to a Dataset object
        dataset_example = Dataset.from_dict(processed_example)

        # Make a prediction
        outputs = trainer.predict(dataset_example)
        prediction = np.argmax(outputs.predictions, axis=1)[0]

        # Print the results
        print(f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}\nActual Label: {example['label']}, Predicted Label: {prediction}\n")


def collate_fn(batch):
    # Convert each item in the batch to a tensor
    processed_batch = {key: torch.tensor([d[key] for d in batch]) for key in batch[0]}
    return processed_batch

def compute_individual_losses_and_confidences(model, dataset, batch_size=16):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    model.eval()

    individual_losses = []
    confidences = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(model.device)

            outputs = model(**inputs)

            # Manually compute the loss
            if 'labels' in batch:
                logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, model.num_labels), labels.view(-1))
                loss_value = loss.item()
            else:
                loss_value = None

            individual_losses.append(loss_value)

            # Compute confidence
            softmax = torch.nn.Softmax(dim=1)
            confidence = softmax(outputs.logits)
            max_confidence, _ = torch.max(confidence, dim=1)
            confidences.extend(max_confidence.cpu().numpy())

    return individual_losses, confidences

def categorize_examples(losses, confidences, threshold):
    categories = []
    for loss, confidence in zip(losses, confidences):
        if loss < threshold['easy'] and confidence > threshold['confidence']:
            categories.append('easy')
        elif loss > threshold['hard']:
            categories.append('hard')
        else:
            categories.append('ambiguous')
    return categories

def main():
    argp = HfArgumentParser(TrainingArguments)
    argp.add_argument('--model', type=str, default='google/electra-small-discriminator', help="Base model to fine-tune.")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True, help="Task to train/evaluate on.")
    argp.add_argument('--dataset', type=str, default=None, help="Overrides default dataset for the task.")
    argp.add_argument('--max_length', type=int, default=128, help="Maximum sequence length.")
    argp.add_argument('--max_train_samples', type=int, default=None, help='Limit number of training examples.')
    argp.add_argument('--max_eval_samples', type=int, default=None, help='Limit number of evaluation examples.')

    training_args, args = argp.parse_args_into_dataclasses()

    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=args.dataset)
        eval_split = 'train'
    else:
        dataset_id = tuple(args.dataset.split(':')) if args.dataset else ('snli',)
        eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
        dataset = load_dataset(*dataset_id)

    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}
    model_class = AutoModelForSequenceClassification if args.task == 'nli' else AutoModelForQuestionAnswering
    model = model_class.from_pretrained(args.model, **task_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    prepare_dataset = prepare_dataset_nli if args.task == 'nli' else prepare_validation_dataset_qa  # Replace with your actual function

    if dataset_id == ('snli',):
        dataset = dataset.filter(lambda ex: ex['label'] != -1)

    train_dataset = dataset['train']
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    train_dataset_featurized = train_dataset.map(lambda ex: prepare_dataset(ex, tokenizer, args.max_length), batched=True, num_proc=NUM_PREPROCESSING_WORKERS, remove_columns=train_dataset.column_names)

    eval_dataset = dataset[eval_split]
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    eval_dataset_featurized = eval_dataset.map(lambda ex: prepare_dataset(ex, tokenizer, args.max_length), batched=True, num_proc=NUM_PREPROCESSING_WORKERS, remove_columns=eval_dataset.column_names)

    # Define the path to the checkpoint directory
    checkpoint_dir = "./trained_model/checkpoint-206000"

    # Check if the checkpoint directory contains any checkpoint
    checkpoint = None
    if os.path.isdir(checkpoint_dir) and any(os.listdir(checkpoint_dir)):
        checkpoint = checkpoint_dir

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset_featurized, 
        eval_dataset=eval_dataset_featurized, 
        tokenizer=tokenizer
    )

    if training_args.do_train:
        # Train the model, resuming from the checkpoint if available
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()


        # Custom loss computation - Make sure batch_size is an integer, e.g., 16
        # Inside main function after training
        losses, confidences = compute_individual_losses_and_confidences(model, train_dataset_featurized, batch_size=16)

        # Define your thresholds based on your dataset and requirement
        threshold = {
            'easy': 0.2,  # Example threshold for easy examples
            'hard': 0.8,  # Example threshold for hard examples
            'confidence': 0.9  # Example confidence threshold
        }

        categories = categorize_examples(losses, confidences, threshold)
        # Creating focused training subsets based on categories
        hard_indices = [i for i, cat in enumerate(categories) if cat == 'hard']
        ambiguous_indices = [i for i, cat in enumerate(categories) if cat == 'ambiguous']
        focused_indices = hard_indices + ambiguous_indices
        focused_train_dataset = train_dataset_featurized.select(focused_indices)

        # Retrain model on focused dataset
        trainer.train_dataset = focused_train_dataset
        trainer.train()

        # After retraining the model
        print("Retraining complete. Computing losses and confidences on the new model...")
        new_losses, new_confidences = compute_individual_losses_and_confidences(model, focused_train_dataset, batch_size=16)

        print("Individual Losses for Focused Subset After Retraining:")
        for i, loss in enumerate(new_losses):
            print(f"Example {i}: Loss = {loss}")

    if training_args.do_eval:
        # Evaluate on the standard dataset
        standard_eval_results = trainer.evaluate()
        print('Standard Evaluation results:')
        print(standard_eval_results)

        # Evaluate and print individual examples for the contrast set
        contrast_set_path = 'contrastset.json'
        contrast_data = load_data_set(contrast_set_path)
        evaluate_and_print_individual_examples(trainer, tokenizer, contrast_data, args, "Contrast")

        # Evaluate and print individual examples for the checklist set
        checklist_set_path = 'checklistset.json'
        checklist_data = load_data_set(checklist_set_path)
        evaluate_and_print_individual_examples(trainer, tokenizer, checklist_data, args, "Checklist")

        # Saving the results (only standard evaluation results)
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(os.path.join(training_args.output_dir, 'standard_eval_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(standard_eval_results, f)

if __name__ == "__main__":
    main()
