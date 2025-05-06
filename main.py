from transformers import AutoProcessor, AutoModelForSeq2SeqLM
import torch
import torchaudio

# Step 1: Load the processor and model
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = AutoModelForSeq2SeqLM.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", torch_dtype=torch.float16, device_map="auto")

# Step 2: Load your audio
waveform, sample_rate = torchaudio.load("/content/sample-0.mp3")
waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
resampled_waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

# Step 3: Preprocess the audio
inputs = processor(audio=resampled_waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").to(model.device)

# Step 4: Transcribe
with torch.no_grad():
    predicted_ids = model.generate(**inputs, max_new_tokens=128)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# Step 5: Print the transcription
print("Transcription:", transcription)
