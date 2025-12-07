"""Test what parameters the encoder actually accepts"""
import torch
from transformers import DetrForObjectDetection

print("Loading model...")
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50', num_labels=7, ignore_mismatched_sizes=True)

# Create dummy inputs
batch_size = 2
seq_len = 100
hidden_dim = 256

feat = torch.randn(batch_size, seq_len, hidden_dim)
pos_emb = torch.randn(batch_size, seq_len, hidden_dim)

print("\nTest 1: encoder with inputs_embeds only")
try:
    out = model.model.encoder(inputs_embeds=feat)
    print("SUCCESS: encoder accepts inputs_embeds")
    print(f"Output shape: {out.last_hidden_state.shape}")
except Exception as e:
    print(f"FAILED: {e}")

print("\nTest 2: encoder with inputs_embeds + position_embeddings")
try:
    out = model.model.encoder(inputs_embeds=feat, position_embeddings=pos_emb)
    print("SUCCESS: encoder accepts inputs_embeds + position_embeddings")
    print(f"Output shape: {out.last_hidden_state.shape}")
except Exception as e:
    print(f"FAILED: {e}")

print("\nTest 3: Check encoder layer parameters")
layer = model.model.encoder.layers[0]
try:
    out = layer(feat)
    print("SUCCESS: encoder layer accepts hidden_states only")
    print(f"Output shape: {out[0].shape}")
except Exception as e:
    print(f"FAILED: {e}")

print("\nTest 4: Check if encoder layer accepts position_embeddings")
try:
    out = layer(feat, position_embeddings=pos_emb)
    print("SUCCESS: encoder layer accepts position_embeddings")
    print(f"Output shape: {out[0].shape}")
except Exception as e:
    print(f"FAILED: {e}")
