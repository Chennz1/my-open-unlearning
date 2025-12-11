"""
GSM-8K Evaluator for Relearn Attack Study
Evaluates model performance on GSM-8K mathematical reasoning tasks
"""
import re
import logging
from typing import Dict, Any
from tqdm import tqdm
import torch
from datasets import load_dataset

from evals.base import Evaluator

logger = logging.getLogger("evaluator")


class GSM8KEvaluator(Evaluator):
    """Evaluator for GSM-8K dataset"""
    
    def __init__(self, eval_cfg, **kwargs):
        super().__init__(name="GSM8K", eval_cfg=eval_cfg, **kwargs)
        self.max_new_tokens = eval_cfg.get("max_new_tokens", 512)
        self.batch_size = eval_cfg.get("batch_size", 4)
        self.subset = eval_cfg.get("subset", "test")  # "train" or "test"
        self.num_samples = eval_cfg.get("num_samples", None)  # None = all samples
        
    def extract_answer(self, text: str) -> str:
        """
        Extract numerical answer from model output
        GSM-8K answers are typically formatted as: #### {answer}
        """
        # Try to find answer after ####
        match = re.search(r'####\s*([+-]?[\d,]+\.?\d*)', text)
        if match:
            return match.group(1).replace(',', '')
        
        # Fallback: try to find last number in text
        numbers = re.findall(r'[+-]?[\d,]+\.?\d*', text)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return ""
    
    def check_answer(self, pred: str, gold: str) -> bool:
        """Check if predicted answer matches gold answer"""
        try:
            pred_num = float(self.extract_answer(pred))
            gold_num = float(self.extract_answer(gold))
            return abs(pred_num - gold_num) < 1e-3
        except (ValueError, TypeError):
            return False
    
    def generate_response(self, model, tokenizer, prompt: str, template_args: Dict) -> str:
        """Generate response for a single prompt"""
        # Format prompt using template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=template_args.get("max_length", 512)
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return response
    
    def evaluate(self, model, tokenizer, template_args, output_dir=None, overwrite=None, **kwargs):
        """Evaluate model on GSM-8K dataset"""
        # Set overwrite flag
        overwrite = self.eval_cfg.overwrite if overwrite is None else overwrite
        
        # Prepare model
        model = self.prepare_model(model)
        
        # Set output directory
        output_dir = output_dir if output_dir else self.eval_cfg.output_dir
        logs_file_path = self.get_logs_file_path(output_dir)
        summary_file_path = self.get_logs_file_path(output_dir, suffix="SUMMARY")
        
        # Load existing results if not overwriting
        logs = self.load_logs_from_file(logs_file_path) if not overwrite else {}
        
        # Check if already evaluated
        if not overwrite and "accuracy" in logs:
            logger.info("GSM-8K evaluation already completed, skipping.")
            summary = self.load_logs_from_file(summary_file_path)
            return summary
        
        logger.info(f"***** Running GSM-8K evaluation on {self.subset} set *****")
        
        # Load dataset
        dataset = load_dataset("openai/gsm8k", "main", split=self.subset)
        if self.num_samples:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))
        
        logger.info(f"Evaluating on {len(dataset)} samples")
        
        # Evaluate
        correct = 0
        total = 0
        predictions = []
        
        for idx, example in enumerate(tqdm(dataset, desc="Evaluating GSM-8K")):
            question = example["question"]
            gold_answer = example["answer"]
            
            try:
                # Generate prediction
                prediction = self.generate_response(model, tokenizer, question, template_args)
                
                # Check correctness
                is_correct = self.check_answer(prediction, gold_answer)
                correct += int(is_correct)
                total += 1
                
                # Store prediction
                predictions.append({
                    "index": idx,
                    "question": question,
                    "gold_answer": gold_answer,
                    "prediction": prediction,
                    "extracted_pred": self.extract_answer(prediction),
                    "extracted_gold": self.extract_answer(gold_answer),
                    "correct": is_correct
                })
                
            except Exception as e:
                logger.warning(f"Error evaluating sample {idx}: {e}")
                predictions.append({
                    "index": idx,
                    "question": question,
                    "gold_answer": gold_answer,
                    "prediction": "",
                    "error": str(e),
                    "correct": False
                })
                total += 1
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0.0
        
        # Save detailed logs
        logs = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "subset": self.subset,
            "predictions": predictions
        }
        self.save_logs(logs, logs_file_path)
        
        # Save summary
        summary = {
            "gsm8k_accuracy": accuracy,
            "gsm8k_correct": correct,
            "gsm8k_total": total
        }
        self.save_logs(summary, summary_file_path)
        
        logger.info(f"GSM-8K Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        return summary
