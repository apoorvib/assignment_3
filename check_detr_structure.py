"""Quick script to check DETR structure"""
import torch
from transformers import DetrForObjectDetection

model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50', num_labels=7, ignore_mismatched_sizes=True)

print("DetrModel attributes:")
print([x for x in dir(model.model) if not x.startswith('_') and not callable(getattr(model.model, x, None))])

if hasattr(model.model, 'encoder'):
    print("\nHas encoder: YES")
if hasattr(model.model, 'decoder'):
    print("Has decoder: YES")
if hasattr(model.model, 'position_embeddings'):
    print("Has position_embeddings: YES")
if hasattr(model.model, 'query_position_embeddings'):
    print("Has query_position_embeddings: YES")

