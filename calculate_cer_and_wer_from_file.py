import argparse
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder.ctc_text_encoder import CTCTextEncoder
from src.utils.io_utils import read_json


def process_files(input_dir):
    total_wer_score = 0
    total_cer_score = 0
    file_counter = 0
    
    json_files = list(Path(input_dir).rglob("*.json"))
    
    if not json_files:
        print("No JSON files found in the specified directory.")
        return
    
    for json_file in json_files:
        file_content = read_json(json_file)
        original_text = file_content.get("text", "")
        normalized_text = CTCTextEncoder.normalize_text(original_text)
        generated_text = file_content.get("pred_text", "")
        
        total_cer_score += calc_cer(normalized_text, generated_text)
        total_wer_score += calc_wer(normalized_text, generated_text)
        file_counter += 1
    
    if file_counter > 0:
        wer_average = total_wer_score / file_counter
        cer_average = total_cer_score / file_counter
        print(f"Average WER across {file_counter} files: {wer_average:.4f}")
        print(f"Average CER across {file_counter} files: {cer_average:.4f}")
    else:
        print("No valid files to process.")


def parse_arguments():
    argument_parser = argparse.ArgumentParser(description="Calculate average WER and CER from predictions.")
    argument_parser.add_argument("--input-dir", required=True, type=str, help="Directory containing JSON files with predictions")
    return argument_parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    process_files(args.input_dir)
