import os
import argparse
import gdown

def download_model(output_path):
    """
    Downloads the fine-tuned Sortformer model from Google Drive.
    """
    file_id = "1c3f8vqnacPRPT1BKC4XyJq2jvG8txo7J"
    url = f"https://drive.google.com/uc?id={file_id}"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(output_path):
        print(f"Model already exists at: {output_path}")
        return output_path
        
    print(f"Downloading model to: {output_path}...")
    try:
        gdown.download(url, output_path, quiet=False)
        print("\nDownload complete! The model is ready for inference.")
    except Exception as e:
        print(f"\nFailed to download model: {e}")
        print("Please ensure you have an active internet connection.")
        print("Also check that 'gdown' is installed: pip install gdown")
        
    return output_path

if __name__ == "__main__":

    default_model_path = os.path.join("models", "speaker_diarization", "sortformer_streaming_4spk_v2.nemo")
    
    parser = argparse.ArgumentParser(description="Download the fine-tuned Sortformer model.")
    parser.add_argument(
        "--output", 
        type=str, 
        default=default_model_path, 
        help="Path where the model will be saved."
    )
    
    args = parser.parse_args()
    download_model(args.output)
