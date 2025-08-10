import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- Configuration ---
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct" # Or another powerful instruction-tuned model

# --- Model Loading ---
def load_model():
    """Loads the quantized model and tokenizer from Hugging Face."""
    
    # Configuration for 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # A pad token is required for batching, but Llama 3 doesn't have one by default.
    # We can use the EOS token for this purpose.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto", # Automatically maps model layers to available devices (GPU/CPU)
        torch_dtype=torch.bfloat16
    )
    return model, tokenizer

def clean_llm_output(raw_output: str) -> dict:
    """Cleans and parses the LLM's JSON output."""
    # The model sometimes wraps the JSON in ```json ... ```
    match = re.search(r"```json\n({.*?})\n```", raw_output, re.S)
    if match:
        json_str = match.group(1)
    else:
        # If no markdown block, assume the JSON starts with {
        json_str = raw_output[raw_output.find('{'):raw_output.rfind('}')+1]
        
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from model output.")
        print("Raw Output:", raw_output)
        return None

# --- Pipeline Functions ---

def build_extraction_prompt(text: str) -> list:
    """Builds the prompt for the SDOH extraction task."""
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
    
    user_prompt = f"Here is the clinical note:\n\n---\n{text}\n---"
    
    # Use the chat template format for Llama 3
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def build_risk_prompt(sdoh_json: dict) -> list:
    """Builds the prompt for the risk categorization task."""
    system_prompt = (
        "You are a clinical risk assessment expert. Based on the provided SDOH findings in JSON format, "
        "evaluate the member's overall SDOH risk level. The risk level must be one of: 'Low', 'Medium', or 'High'.\n\n"
        "Provide a final assessment in a structured JSON format with the keys: "
        "'overall_sdoh_risk', 'risk_score' (an integer from 0 to 100), and 'justification'."
    )
    
    user_prompt = f"Here are the extracted SDOH findings:\n\n```json\n{json.dumps(sdoh_json, indent=2)}\n```"
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def run_inference(model, tokenizer, prompt_messages: list) -> dict:
    """Runs a single inference pass and returns cleaned JSON."""
    # Apply the chat template
    input_ids = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate the response
    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    
    response_ids = outputs[0][input_ids.shape[-1]:]
    raw_output = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    return clean_llm_output(raw_output)


# --- Main Execution ---
if __name__ == "__main__":
    print("Loading model... (This may take a few minutes)")
    model, tokenizer = load_model()
    print("Model loaded successfully.")

    sample_note = (
        "Patient ID: 98765. Date: 2025-08-10. "
        "Ms. Garcia is a 68 y/o female presenting for follow-up. "
        "She reports significant stress about making rent this month after her part-time "
        "job at the diner cut her hours back. She admits that near the end of the month, she "
        "sometimes has to skip a meal to afford her medications. She lives alone and "
        "mentions feeling lonely since her son, who used to drive her to appointments, "
        "moved to another state last year. Getting here today was difficult as the bus was late."
    )
    
    # --- Step 1 & 2: Extract SDOH ---
    print("\n--- Running SDOH Extraction ---")
    extraction_prompt = build_extraction_prompt(sample_note)
    extracted_sdoh = run_inference(model, tokenizer, extraction_prompt)
    
    if extracted_sdoh:
        print("Extracted SDOH Data:")
        print(json.dumps(extracted_sdoh, indent=2))
        
        # --- Step 3: Categorize Risk ---
        print("\n--- Running Risk Categorization ---")
        risk_prompt = build_risk_prompt(extracted_sdoh)
        risk_assessment = run_inference(model, tokenizer, risk_prompt)
        
        if risk_assessment:
            print("Risk Assessment:")
            print(json.dumps(risk_assessment, indent=2))