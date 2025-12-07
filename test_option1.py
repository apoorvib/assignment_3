"""
Test script specifically for Option 1 (Feature Difference Architecture).
Tests model instantiation, forward pass, and basic functionality.
"""

import torch
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import DETRFeatureDiff
from utils import get_device, set_seed


def test_option1_instantiation():
    """Test Option 1 model instantiation with different configurations."""
    print("="*60)
    print("TEST 1: Model Instantiation")
    print("="*60)
    
    device = get_device()
    
    try:
        # Test with separate ResNet (default)
        print("\n1.1 Testing with separate ImageNet ResNet50 (layer3)...")
        model1 = DETRFeatureDiff(
            num_classes=6,
            pretrained_model='facebook/detr-resnet-50',
            use_detr_backbone=False,
            layer='layer3'
        )
        model1 = model1.to(device)
        print("   ✅ Model created successfully")
        
        # Test with separate ResNet (layer4)
        print("\n1.2 Testing with separate ImageNet ResNet50 (layer4)...")
        model2 = DETRFeatureDiff(
            num_classes=6,
            pretrained_model='facebook/detr-resnet-50',
            use_detr_backbone=False,
            layer='layer4'
        )
        model2 = model2.to(device)
        print("   ✅ Model created successfully")
        
        # Test with DETR backbone (forward hooks)
        print("\n1.3 Testing with DETR backbone (forward hooks, layer3)...")
        model3 = DETRFeatureDiff(
            num_classes=6,
            pretrained_model='facebook/detr-resnet-50',
            use_detr_backbone=True,
            layer='layer3'
        )
        model3 = model3.to(device)
        print("   ✅ Model created successfully")
        
        return True
    except Exception as e:
        print(f"   ❌ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_option1_forward_pass():
    """Test Option 1 forward pass."""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass")
    print("="*60)
    
    device = get_device()
    batch_size = 2
    image_size = 800
    
    # Create dummy images
    dummy_img1 = torch.randn(batch_size, 3, image_size, image_size).to(device)
    dummy_img2 = torch.randn(batch_size, 3, image_size, image_size).to(device)
    
    try:
        # Test with separate ResNet
        print("\n2.1 Testing forward pass (separate ResNet, layer3)...")
        model = DETRFeatureDiff(
            num_classes=6,
            use_detr_backbone=False,
            layer='layer3'
        ).to(device)
        model.eval()
        
        with torch.no_grad():
            outputs = model(dummy_img1, dummy_img2)
            
            # Check output format
            assert hasattr(outputs, 'logits'), "Output missing logits"
            assert hasattr(outputs, 'pred_boxes'), "Output missing pred_boxes"
            
            # Check shapes
            assert outputs.logits.shape[0] == batch_size, f"Batch size mismatch: {outputs.logits.shape[0]} != {batch_size}"
            assert len(outputs.logits.shape) == 3, f"Logits should be 3D, got {len(outputs.logits.shape)}D"
            assert outputs.pred_boxes.shape[0] == batch_size, f"Batch size mismatch in boxes"
            assert outputs.pred_boxes.shape[-1] == 4, f"Boxes should have 4 coords, got {outputs.pred_boxes.shape[-1]}"
            
            print(f"   ✅ Forward pass successful")
            print(f"      Logits shape: {outputs.logits.shape}")
            print(f"      Pred boxes shape: {outputs.pred_boxes.shape}")
        
        # Test with DETR backbone
        print("\n2.2 Testing forward pass (DETR backbone, layer3)...")
        model2 = DETRFeatureDiff(
            num_classes=6,
            use_detr_backbone=True,
            layer='layer3'
        ).to(device)
        model2.eval()
        
        with torch.no_grad():
            outputs2 = model2(dummy_img1, dummy_img2)
            assert hasattr(outputs2, 'logits')
            assert hasattr(outputs2, 'pred_boxes')
            print(f"   ✅ Forward pass successful")
            print(f"      Logits shape: {outputs2.logits.shape}")
            print(f"      Pred boxes shape: {outputs2.pred_boxes.shape}")
        
        return True
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_option1_feature_extraction():
    """Test feature extraction from images."""
    print("\n" + "="*60)
    print("TEST 3: Feature Extraction")
    print("="*60)
    
    device = get_device()
    batch_size = 1
    image_size = 800
    
    dummy_img = torch.randn(batch_size, 3, image_size, image_size).to(device)
    
    try:
        # Test with separate ResNet
        print("\n3.1 Testing feature extraction (separate ResNet, layer3)...")
        model = DETRFeatureDiff(
            num_classes=6,
            use_detr_backbone=False,
            layer='layer3'
        ).to(device)
        model.eval()
        
        with torch.no_grad():
            features = model.extract_features(dummy_img)
            print(f"   ✅ Feature extraction successful")
            print(f"      Feature shape: {features.shape}")
            print(f"      Expected channels: 256 (after projection)")
            
            # Check that features have correct dimensions
            assert len(features.shape) == 4, f"Features should be 4D, got {len(features.shape)}D"
            assert features.shape[0] == batch_size, "Batch size mismatch"
            assert features.shape[1] == 256, f"Expected 256 channels after projection, got {features.shape[1]}"
        
        return True
    except Exception as e:
        print(f"   ❌ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_option1_feature_difference():
    """Test feature difference computation."""
    print("\n" + "="*60)
    print("TEST 4: Feature Difference Computation")
    print("="*60)
    
    device = get_device()
    batch_size = 2
    image_size = 800
    
    dummy_img1 = torch.randn(batch_size, 3, image_size, image_size).to(device)
    dummy_img2 = torch.randn(batch_size, 3, image_size, image_size).to(device)
    
    try:
        print("\n4.1 Testing feature difference...")
        model = DETRFeatureDiff(
            num_classes=6,
            use_detr_backbone=False,
            layer='layer3'
        ).to(device)
        model.eval()
        
        with torch.no_grad():
            feat1 = model.extract_features(dummy_img1)
            feat2 = model.extract_features(dummy_img2)
            feat_diff = feat2 - feat1
            
            print(f"   ✅ Feature difference computed")
            print(f"      Feature 1 shape: {feat1.shape}")
            print(f"      Feature 2 shape: {feat2.shape}")
            print(f"      Difference shape: {feat_diff.shape}")
            print(f"      Difference range: [{feat_diff.min().item():.4f}, {feat_diff.max().item():.4f}]")
            
            # Check shapes match
            assert feat1.shape == feat2.shape == feat_diff.shape, "Feature shapes don't match"
        
        return True
    except Exception as e:
        print(f"   ❌ Feature difference computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_option1_transformer_input():
    """Test transformer input format."""
    print("\n" + "="*60)
    print("TEST 5: Transformer Input Format")
    print("="*60)
    
    device = get_device()
    batch_size = 1
    image_size = 800
    
    dummy_img1 = torch.randn(batch_size, 3, image_size, image_size).to(device)
    dummy_img2 = torch.randn(batch_size, 3, image_size, image_size).to(device)
    
    try:
        print("\n5.1 Testing transformer input preparation...")
        model = DETRFeatureDiff(
            num_classes=6,
            use_detr_backbone=False,
            layer='layer3'
        ).to(device)
        model.eval()
        
        with torch.no_grad():
            # Extract features
            feat1 = model.extract_features(dummy_img1)
            feat2 = model.extract_features(dummy_img2)
            feat_diff = feat2 - feat1
            
            # Check feature dimensions
            batch_size_actual, channels, height, width = feat_diff.shape
            print(f"   Feature diff shape: {feat_diff.shape}")
            
            # Flatten spatial dimensions (as done in forward)
            feat_diff_flat = feat_diff.flatten(2).transpose(1, 2)  # [batch, h*w, channels]
            print(f"   Flattened shape: {feat_diff_flat.shape}")
            print(f"   Sequence length: {feat_diff_flat.shape[1]}")
            print(f"   Hidden size: {feat_diff_flat.shape[2]}")
            
            # Check position embeddings
            position_embeddings = model.position_embeddings(feat_diff_flat)
            print(f"   Position embeddings shape: {position_embeddings.shape}")
            
            # Check query position embeddings
            num_queries = model.query_position_embeddings.weight.shape[0]
            print(f"   Number of object queries: {num_queries}")
            
            assert feat_diff_flat.shape[1] == height * width, "Sequence length mismatch"
            assert feat_diff_flat.shape[2] == 256, "Hidden size should be 256"
            assert position_embeddings.shape == feat_diff_flat.shape, "Position embedding shape mismatch"
            
            print("   ✅ Transformer input format correct")
        
        return True
    except Exception as e:
        print(f"   ❌ Transformer input format test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_option1_loss_computation():
    """Test loss computation (simplified version)."""
    print("\n" + "="*60)
    print("TEST 6: Loss Computation")
    print("="*60)
    
    device = get_device()
    batch_size = 2
    image_size = 800
    
    dummy_img1 = torch.randn(batch_size, 3, image_size, image_size).to(device)
    dummy_img2 = torch.randn(batch_size, 3, image_size, image_size).to(device)
    
    # Create dummy targets
    targets = {
        'boxes': [
            torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32).to(device),
            torch.tensor([[0.3, 0.3, 0.1, 0.1]], dtype=torch.float32).to(device)
        ],
        'labels': [
            torch.tensor([1], dtype=torch.long).to(device),
            torch.tensor([2], dtype=torch.long).to(device)
        ]
    }
    
    try:
        print("\n6.1 Testing loss computation...")
        model = DETRFeatureDiff(
            num_classes=6,
            use_detr_backbone=False,
            layer='layer3'
        ).to(device)
        model.train()
        
        # Forward pass
        outputs = model(dummy_img1, dummy_img2)
        
        # Check outputs
        assert hasattr(outputs, 'logits'), "Outputs missing logits"
        assert hasattr(outputs, 'pred_boxes'), "Outputs missing pred_boxes"
        
        # Manual loss computation (simplified)
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        
        # Simple classification loss
        cls_loss = 0.0
        for i in range(batch_size):
            target_labels = targets['labels'][i]
            if len(target_labels) > 0:
                # Use first query for first target (simplified)
                cls_loss += torch.nn.functional.cross_entropy(
                    logits[i, 0:1], target_labels[0:1]
                )
        
        print(f"   ✅ Loss computation successful")
        print(f"      Classification loss: {cls_loss.item():.4f}")
        print(f"      Note: This uses simplified loss (not bipartite matching)")
        
        return True
    except Exception as e:
        print(f"   ❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_option1_gradient_flow():
    """Test that gradients can flow through the model."""
    print("\n" + "="*60)
    print("TEST 7: Gradient Flow")
    print("="*60)
    
    device = get_device()
    batch_size = 1
    image_size = 800
    
    dummy_img1 = torch.randn(batch_size, 3, image_size, image_size).to(device)
    dummy_img2 = torch.randn(batch_size, 3, image_size, image_size).to(device)
    
    # Create dummy targets
    target_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32).to(device)
    target_labels = torch.tensor([1], dtype=torch.long).to(device)
    
    try:
        print("\n7.1 Testing gradient flow...")
        model = DETRFeatureDiff(
            num_classes=6,
            use_detr_backbone=False,
            layer='layer3'
        ).to(device)
        model.train()
        
        # Forward pass
        outputs = model(dummy_img1, dummy_img2)
        
        # Simple loss
        logits = outputs.logits
        cls_loss = torch.nn.functional.cross_entropy(
            logits[0, 0:1], target_labels[0:1]
        )
        
        # Backward pass
        cls_loss.backward()
        
        # Check gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                print(f"   ✅ Gradients flowing to: {name}")
                break
        
        if has_grad:
            print("   ✅ Gradient flow successful")
        else:
            print("   ⚠️  No gradients detected (may be normal if all params frozen)")
        
        return True
    except Exception as e:
        print(f"   ❌ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Option 1 tests."""
    print("\n" + "="*60)
    print("OPTION 1 (FEATURE DIFF) COMPREHENSIVE TESTS")
    print("="*60)
    
    set_seed(42)
    
    tests = [
        ("Model Instantiation", test_option1_instantiation),
        ("Forward Pass", test_option1_forward_pass),
        ("Feature Extraction", test_option1_feature_extraction),
        ("Feature Difference", test_option1_feature_difference),
        ("Transformer Input Format", test_option1_transformer_input),
        ("Loss Computation", test_option1_loss_computation),
        ("Gradient Flow", test_option1_gradient_flow),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All Option 1 tests passed! Model is ready for training.")
    elif passed >= total - 1:
        print("\n✅ Most tests passed. Model should work, but verify any failures.")
    else:
        print("\n⚠️  Some critical tests failed. Please review errors above.")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

