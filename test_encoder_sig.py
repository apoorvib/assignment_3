import sys
print("Starting...", flush=True)
sys.stdout.flush()

from transformers import DetrForObjectDetection
import inspect

print("Loading model...", flush=True)
sys.stdout.flush()

model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50', num_labels=7, ignore_mismatched_sizes=True)

print("\n=== Encoder forward signature ===", flush=True)
sig = inspect.signature(model.model.encoder.forward)
print(sig, flush=True)
print("\nParameters:", flush=True)
for name, param in sig.parameters.items():
    print(f"  {name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'no annotation'}", flush=True)

print("\n=== Encoder layer forward signature ===", flush=True)
layer = model.model.encoder.layers[0]
sig = inspect.signature(layer.forward)
print(sig, flush=True)
print("\nParameters:", flush=True)
for name, param in sig.parameters.items():
    print(f"  {name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'no annotation'}", flush=True)

print("\n=== Self attention forward signature ===", flush=True)
sig = inspect.signature(layer.self_attn.forward)
print(sig, flush=True)
print("\nParameters:", flush=True)
for name, param in sig.parameters.items():
    print(f"  {name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'no annotation'}", flush=True)

sys.stdout.flush()
