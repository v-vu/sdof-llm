import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer

# --- Configuration ---
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
NEW_ADAPTER_NAME = "llama3-8b-sdoh-dpo-adapter"

# --- 1. Prepare the Preference Dataset ---
# In a real project, this would be loaded from a file (e.g., json, csv)
# This dataset is created by human annotators choosing the 'better' response.
def get_preference_dataset():
    # The 'prompt' should contain the full instruction given to the model
    system_prompt = (
        "You are a helpful, respectful, and honest healthcare data analyst. "
        "Your task is to extract Social Determinants of Health (SDOH) from a clinical note. "
        "Identify evidence for the following categories: "
        "'Food Insecurity', 'Housing Instability', 'Transportation Barriers', "
        "'Financial Strain', 'Social Isolation', 'Employment Issues'.\n\n"
        "Please output your findings in a structured JSON format with a key 'sdoh_findings' "
        "containing a list of objects. Each object must have 'category', 'evidence_text', "
        "and 'certainty_score' (from 0.0 to 1.0)."
    )
    
    # Each item is a dict with 'prompt', 'chosen', and 'rejected'
    data = [
        {
            "prompt": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nPatient reports 'trouble with the landlord' and that the 'cupboards are pretty bare by the end of the week'. Son, who used to drive her, moved away.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "chosen": """```json
{
  "sdoh_findings": [
    {
      "category": "Housing Instability",
      "evidence_text": "trouble with the landlord",
      "certainty_score": 0.95
    },
    {
      "category": "Food Insecurity",
      "evidence_text": "cupboards are pretty bare by the end of the week",
      "certainty_score": 0.98
    },
    {
      "category": "Transportation Barriers",
      "evidence_text": "Son, who used to drive her, moved away",
      "certainty_score": 0.85
    }
  ]
}
```""", # Chosen: More complete and accurate
            "rejected": """```json
{
  "sdoh_findings": [
    {
      "category": "Financial Strain",
      "evidence_text": "trouble with the landlord",
      "certainty_score": 0.90
    }
  ]
}
```""" # Rejected: Incomplete and miscategorized
        },
        # Add more preference pairs here...
    ]
    return Dataset.from_list(data)

# --- 2. Setup Model, Tokenizer, and Trainer ---
if __name__ == "__main__":
    # Load the dataset
    train_dataset = get_preference_dataset()
    
    # Load model and tokenizer with quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA config for efficient training
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
    )

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=50, # Set a small number for demonstration
        learning_rate=2e-4,
        output_dir="./dpo_results",
        logging_steps=10,
        save_strategy="no", # We'll save manually at the end
        report_to="none",
    )

    # DPO Trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None, # The trainer will create a reference model copy automatically
        args=training_args,
        beta=0.1, # The DPO hyperparameter
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=1024,
        max_length=1536,
    )

    # --- 3. Run Training ---
    print("Starting DPO fine-tuning...")
    dpo_trainer.train()
    print("Training complete.")

    # --- 4. Save the Adapter ---
    print(f"Saving LoRA adapter to ./{NEW_ADAPTER_NAME}")
    dpo_trainer.model.save_pretrained(NEW_ADAPTER_NAME)

    # To use this later, you would load the base model and then apply the adapter:
    # from peft import PeftModel
    # base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, ...)
    # tuned_model = PeftModel.from_pretrained(base_model, NEW_ADAPTER_NAME)
    print("\nFine-tuning finished. The new adapter is ready to be used for inference.")