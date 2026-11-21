from datasets import Audio, load_dataset

print("Connecting to the AMI dataset (streaming, nothing big is downloaded)...")
ds = load_dataset("edinburghcstr/ami", "ihm", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))      # keep the raw audio bytes

for i, ex in enumerate(ds):
    audio = ex["audio"]
    size = len(audio["bytes"]) if audio.get("bytes") else 0
    duration = ex["end_time"] - ex["begin_time"]
    print(f'{i}  {ex["meeting_id"]}  {ex["speaker_id"]}  {duration:4.1f}s  '
          f'{size:7d} bytes  | {ex["text"][:70]}')
    if i >= 4:
        break
print("OK")