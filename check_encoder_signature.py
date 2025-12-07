from transformers import DetrForObjectDetection
import inspect

model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50', num_labels=7, ignore_mismatched_sizes=True)

print("Encoder forward signature:")
print(inspect.signature(model.model.encoder.forward))

print("\nEncoder layer forward signature:")
layer = model.model.encoder.layers[0]
print(inspect.signature(layer.forward))

print("\nSelf attention forward signature:")
print(inspect.signature(layer.self_attn.forward))
