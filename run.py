#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.
import os
import re
import argparse
import logging
import math
from random import choice

import datasets
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from models.unified.prefixtuning import Model as T5PrefixTuning
from parent import parent_score

logger = logging.getLogger(__name__)

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def convert_text(text):
    #return text
    text = text.lower()
    text = ' '.join(re.split('(\W)', text))
    text = ' '.join(text.split())
    return text

class Bert:
    def __init__(self, location):
        self.location = location

class PrefixTuning:
    def __init__(self, prefix_sequence_length):
        self.prefix_sequence_length = prefix_sequence_length
        self.mid_dim = 512
        self.prefix_dropout = 0.0

class Model:
    def __init__(self):
        self.use_description = False
        self.knowledge_usage = "concatenate"
        self.freeze_plm = True
        self.freeze_prefix = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="webnlg",
        choices=["webnlg", "dart"],
        help="The name of the dataset to use.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--valid_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the validation data (also the data for generation if --do_gen is passed)."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " 
         "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=512,
        help="The maximum total input sequence length after "
         "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default='paraphrase: ',
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=512,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--input_column",
        type=str,
        default="input",
        help="The name of the column in the datasets containing the inputs.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="ref",
        help="The name of the column in the datasets containing the targets.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3, 
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", 
        type=int, 
        default=0, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=100, 
        help="A random seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--use_sacrebleu",
        action="store_true",
        help="Use sacrebleu for evaluation."
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help='Log training information every N steps'
    )
    parser.add_argument(
        "--no_progress_bar",
        action="store_true",
        help="If passed, will log information of the training progress regularly instead of showing a progress bar"
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="If passed, only do evaluation"
    )
    parser.add_argument(
        "--do_gen",
        action="store_true",
        help="If passed, only do generation"
    )
    parser.add_argument(
        "--eval_after_each_epoch",
        type=int,
        default=1,
        help="Do evaluation when one epoch training ends."
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=0,
        help="Evaluate the model every N steps"
    )
    parser.add_argument(
        "--add_special_tokens",
        action="store_true",
        help="Add <H>, <R>, <T> tokens for graph translation"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=15,
        help="The patience for early stopping"
    )
    parser.add_argument(
        "--prefix_tuning",
        action="store_true",
        help="If set, do prefix tuning instead of fine-tuning"
    )
    parser.add_argument(
        "--prefix_sequence_length",
        type=int,
        default=10,
        help="The length of prefix sequence for prefix tuning"
    )
    parser.add_argument(
        "--not_use_existing_ref",
        action="store_true",
        help="If set, write reference file instead of using existing ones for evaluation"
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.valid_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.valid_file is not None:
            extension = args.valid_file.split(".")[-1]
            assert extension in ["csv", "json"], "valid_file` should be a csv or a json file."

    if args.prefix_tuning:
        args.bert = Bert(args.model_name_or_path)
        args.prefix_tuning = PrefixTuning(args.prefix_sequence_length)
        args.model = Model()

    return args

# class for early stopping
class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = -float("inf")

    def __call__(self, score=None):
        if score != None:
            if score < self.best_score:
                self.counter += 1
            else:
                self.best_score = score
                self.counter = 0
        return self.counter == self.patience

def train(args, logger, datasets, tokenizer, model, config, accelerator):

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    dataset = datasets["train"]
    dataloader = DataLoader(
        dataset, 
        shuffle=True, 
        collate_fn=data_collator, 
        batch_size=args.per_device_train_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    optimizer, model, dataloader = accelerator.prepare(optimizer, model, dataloader)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    def save_checkpoint(checkpoint_dir, model):
        if args.output_dir is not None:
            save_dir = "{}/{}".format(args.output_dir, checkpoint_dir)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)

            # disable logging in save_pretrained function
            transformers.utils.logging.set_verbosity_error()

            unwrapped_model.save_pretrained(
                save_dir, 
                save_function=accelerator.save,
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(save_dir)

            # enable logging again
            if accelerator.is_local_main_process:
                transformers.utils.logging.set_verbosity_info()

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if not args.no_progress_bar:
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    tr_loss, logging_loss = 0., 0.
    best_score = 0.
    early_stopping = EarlyStopping(args.early_stopping_patience)
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(dataloader):
            outputs = model(**batch)

            if args.prefix_tuning:
                loss = outputs["loss"]
            else:
                loss = outputs.loss

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
                if not args.no_progress_bar:
                    progress_bar.update(1)

                if completed_steps % args.logging_steps == 0:
                    cur_loss = (tr_loss - logging_loss) / args.logging_steps
                    if args.no_progress_bar:
                        logger.info(f"  Epoch={epoch+1}, step={completed_steps}, loss={cur_loss:.4f}")
                    else:
                        progress_bar.set_postfix(
                            {"loss": cur_loss}
                        )
                    logging_loss = tr_loss

                if args.eval_steps > 0 and completed_steps % args.eval_steps == 0:
                    score = eval(
                        args, 
                        logger, 
                        datasets, 
                        tokenizer, 
                        accelerator.unwrap_model(model), 
                        config, 
                        accelerator
                    )

                    if score >= best_score:
                        best_score = score
                        save_checkpoint("checkpoint_best", model)
                    save_checkpoint("checkpoint_last", model)

                    logger.info(f"  Evaluation after step {completed_steps}: score={score:.2f}, best score={best_score:.2f}")

                    if early_stopping(score):
                        break

            if completed_steps >= args.max_train_steps:
                break
        
        save_checkpoint("checkpoint_last", model)
        if args.eval_after_each_epoch:
            score = eval(
                args, 
                logger, 
                datasets, 
                tokenizer, 
                accelerator.unwrap_model(model), 
                config, 
                accelerator
            )

            if score >= best_score:
                best_score = score
                save_checkpoint("checkpoint_best", model)

            logger.info(f"  Evaluation after epoch {epoch+1}: score={score:.2f}, best score={best_score:.2f}")

            if early_stopping(score):
                break

def eval(args, logger, datasets, tokenizer, model, config, accelerator):

    data_collator = DataCollatorWithPadding(
        tokenizer,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    dataset = datasets["valid"]
    dataset, refs = dataset["dataset"], dataset["refs"]
    if not args.use_sacrebleu:
        refs, triples = refs["refs"], refs["triples"]
    dataloader = DataLoader(
        dataset, 
        collate_fn=data_collator, 
        batch_size=args.per_device_eval_batch_size,
    )
    model, dataloader = accelerator.prepare(model, dataloader)

    # Metric
    if args.use_sacrebleu:
        metric = load_metric("sacrebleu")
    else:
        metric = None

    # Evaluate!
    model.eval()

    if args.val_max_target_length is not None:
        max_length = args.val_max_target_length
    elif args.max_target_length is not None:
        max_length = args.max_target_length
    else:
        max_length = config.max_length

    gen_kwargs = {
        "max_length": max_length,
        "num_beams": args.num_beams,
    }

    if args.do_eval:
        logger.info("***** Running evaluation *****")
        logger.info(f"  Num examples = {len(dataset)}")
        logger.info(f"  Total evaluation batch size = {args.per_device_eval_batch_size}")
    output_preds = []
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            output_preds += decoded_preds

    # The accelerator dataloader will return the first few items in the dataset
    # at the end of the epoch to keep the batches on the same size on all GPUs
    # Thus we need to truncate the output
    output_preds = output_preds[:len(refs)]

    if metric is not None:
        metric.add_batch(predictions=output_preds, references=refs)
        score = metric.compute()["score"]
        if args.do_eval: 
            logger.info(f"  Evaluation results: score={score:.2f}")
    # TODO: not sure its behavior in multi-processing
    else:
        dir_path = os.path.dirname(os.path.realpath(__file__)) + "/data"
        pred_path = f"{args.output_dir}/predictions.txt"
        with open(pred_path, "w", encoding="utf-8") as f:
            f.writelines(convert_text(s) + "\n" for s in output_preds)

        split = re.findall(r".*/(.*).json", args.valid_file)[0]
        if args.not_use_existing_ref:
            file_prefix = f"{args.output_dir}/{split}.target"
        else:
            file_prefix = os.path.dirname(os.path.realpath(__file__)) + f"/ref_data/{args.dataset_name}/{split}.target"

        if args.not_use_existing_ref:
            # multi-bleu.perl for evaluation
            with open(f"{file_prefix}_eval", "w") as f1, open(f"{file_prefix}2_eval", "w") as f2, open(f"{file_prefix}3_eval", "w") as f3:
                for flag, ref in enumerate(refs):
                    if flag > 0:
                        text_prefix = "\n"
                    else:
                        text_prefix = ""
                    for i, f in enumerate([f1, f2, f3]):
                        if i < len(ref):
                            text = text_prefix + convert_text(ref[i])
                        f.write(text)

        cmd_string = f"perl {dir_path}/multi-bleu.perl -lc " \
                    + f"{file_prefix}_eval {file_prefix}2_eval {file_prefix}3_eval " \
                    + f"< {pred_path}"

        res = os.popen(cmd_string).read()
        score = float(re.findall(r"BLEU = (.*), .*\(.*\)", res)[0])

        if args.not_use_existing_ref:
            os.remove(f"{file_prefix}_eval")
            os.remove(f"{file_prefix}2_eval")
            os.remove(f"{file_prefix}3_eval")

        if args.do_eval:
            # meteor
            if args.not_use_existing_ref:
                text_prefix = ""
                with open(f"{file_prefix}_eval_meteor", "w") as f:
                    for flag, ref in enumerate(refs):
                        for i in range(3):
                            if i < len(ref):
                                text = text_prefix + convert_text(ref[i])
                            f.write(text)
                            text_prefix = "\n"

            cmd_string = f"java -jar {dir_path}/utils/meteor-1.5.jar {pred_path} " \
                        + f"{file_prefix}_eval_meteor -l en -norm -r 3"
            res = os.popen(cmd_string).read()
            meteor_score = float(res.split()[-1]) * 100

            if args.not_use_existing_ref:
                os.remove(f"{file_prefix}_eval_meteor")

            # parent
            _, _, f1 = parent_score(output_preds, refs, triples)

            logger.info(f"  Evaluation results: BLEU={score:.2f}, METEOR={meteor_score:.2f}, PARENT-F1={100*f1:.2f}")

        os.remove(pred_path)
    return score


def gen(args, logger, datasets, tokenizer, model, config, accelerator):

    data_collator = DataCollatorWithPadding(
        tokenizer,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    dataset, refs = datasets["valid"]["dataset"], datasets["valid"]["refs"]
    if not args.use_sacrebleu:
        refs, triples = refs["refs"], refs["triples"]
    dataloader = DataLoader(
        dataset, 
        collate_fn=data_collator, 
        batch_size=args.per_device_eval_batch_size,
    )
    model, dataloader = accelerator.prepare(model, dataloader)

    # Generate!
    model.eval()

    if args.val_max_target_length is not None:
        max_length = args.val_max_target_length
    elif args.max_target_length is not None:
        max_length = args.max_target_length
    else:
        max_length = config.max_length

    gen_kwargs = {
        "max_length": max_length,
        "num_beams": args.num_beams,
    }

    logger.info("***** Running generation *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Total evaluation batch size = {args.per_device_eval_batch_size * accelerator.num_processes}")
    output_preds = []
    for step, batch in enumerate(dataloader):
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            output_preds += decoded_preds

    # The accelerator dataloader will return the first few items in the dataset
    # at the end of the epoch to keep the batches on the same size on all GPUs
    # Thus we need to truncate the output
    output_preds = output_preds[:len(refs)]

    with open(f"{args.output_dir}/gen_output.txt", "w", encoding="utf-8") as f:
        for pred in output_preds:
            f.write(pred + "\n")

    logger.info(f"  Generation finished!")



def main(args):

    args.do_train = not (args.do_eval or args.do_gen)
    if args.do_train:
        args.use_sacrebleu = True

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()

    logging_handlers = []
    if accelerator.is_main_process and args.output_dir is not None:
        if args.do_train:
            os.makedirs(args.output_dir + "/checkpoint_best", exist_ok=True)
            os.makedirs(args.output_dir + "/checkpoint_last", exist_ok=True)
            logging_handlers.append(logging.FileHandler(os.path.join(args.output_dir, 'train.log')))
        else:
            os.makedirs(args.output_dir, exist_ok=True)
            if args.do_eval:
                logging_handlers.append(logging.FileHandler(os.path.join(args.output_dir, 'eval.log')))


    # Make one log on every process with the configuration for debugging.
    logging_handlers.append(logging.StreamHandler())
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=logging_handlers,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info(args)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, 
            use_fast=not args.use_slow_tokenizer,
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, 
            use_fast=not args.use_slow_tokenizer,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. "
            "This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, "
            "using --tokenizer_name."
        )
    
    if args.add_special_tokens:
        new_tokens = ['<H>', '<R>', '<T>']
        new_tokens_vocab = {}
        new_tokens_vocab['additional_special_tokens'] = []
        for idx, t in enumerate(new_tokens):
            new_tokens_vocab['additional_special_tokens'].append(t)
        num_added_toks = tokenizer.add_special_tokens(new_tokens_vocab)

    if args.model_name_or_path:
        # prefixtuning
        if args.prefix_tuning:
            model = T5PrefixTuning(args)
            if not args.do_train:
                state_dict = torch.load(os.path.join(args.model_name_or_path, transformers.WEIGHTS_NAME), map_location="cpu")
                model.load_state_dict(state_dict, strict=True)
                # release memory
                del state_dict
        # finetuning
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    '''
    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )
    '''

    prefix = args.source_prefix if args.source_prefix is not None and not args.prefix_tuning else ""

    extension = args.valid_file.split(".")[-1]
    raw_datasets = {}
    if args.do_train and args.train_file is not None:
        raw_datasets["train"] = load_dataset(extension, data_files={"data": args.train_file})
    # validation set is multi-references so we don't load it together with training set
    if args.valid_file is not None:
        raw_datasets["valid"] = load_dataset(extension, data_files={"data": args.valid_file})

    # Preprocessing the datasets.
    column_names_to_remove = raw_datasets["valid"]["data"].column_names

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[args.input_column]
        targets = examples[args.target_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs, 
            max_length=args.max_source_length, 
            padding=padding, 
            truncation=True,
        )

        # multiple references
        if type(targets[0]) is not list:
            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets, 
                    max_length=max_target_length, 
                    padding=padding, 
                    truncation=True,
                )

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    refs = [sample["ref"] for sample in raw_datasets["valid"]["data"]]
    # process for sacrebleu
    if args.use_sacrebleu:
        max_num_ref = max([len(ref) for ref in refs])
        refs = [(ref if type(ref) is list else [ref]) + [""] * (max_num_ref - len(ref)) for ref in refs]
    # save both refs and triples for PARENT metric computation
    else:
        triples = [sample["triple"] for sample in raw_datasets["valid"]["data"]]
        refs = {"refs":refs, "triples": triples}

    with accelerator.main_process_first():
        processed_datasets = {}
        for split, dataset in raw_datasets.items():
            processed_datasets[split] = dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=column_names_to_remove,
                load_from_cache_file=False,
                desc="Running tokenizer on {} dataset".format(split),
            )

    if "train" in processed_datasets:
        processed_datasets["train"] = processed_datasets["train"]["data"]
    processed_datasets["valid"] = {"dataset": processed_datasets["valid"]["data"], "refs": refs}

    if args.do_train:
        train(args, logger, processed_datasets, tokenizer, model, config, accelerator)
    elif args.do_eval:
        eval(args, logger, processed_datasets, tokenizer, model, config, accelerator)
    else:
        gen(args, logger, processed_datasets, tokenizer, model, config, accelerator)


if __name__ == "__main__":
    main(parse_args())
