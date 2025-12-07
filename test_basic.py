"""
Basic test script to verify the implementation works.
Tests data loading, model instantiation, and forward pass.
"""

import warnings
import os
import sys

# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress specific warning categories
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Suppress specific library warnings
warnings.filterwarnings('ignore', module='transformers')
warnings.filterwarnings('ignore', module='torch')
warnings.filterwarnings('ignore', module='timm')

# Suppress transformers warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
# Suppress PyTorch warnings
torch.set_warn_always(False)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader import MovedObjectDataset, collate_fn
from models import DETRPixelDiff, DETRFeatureDiff
from utils import get_device, set_seed
from torch.utils.data import DataLoader


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from transformers import DetrForObjectDetection
        print("[PASS] All imports successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False


def test_model_instantiation():
    """Test that models can be instantiated."""
    print("\nTesting model instantiation...")
    device = get_device()
    
    try:
        # Test Option 2 (pixel diff)
        print("  Testing Option 2 (Pixel Diff)...")
        model2 = DETRPixelDiff(num_classes=6)
        model2 = model2.to(device)
        print("  [PASS] Option 2 model created successfully")
        
        # Test Option 1 (feature diff)
        print("  Testing Option 1 (Feature Diff)...")
        model1 = DETRFeatureDiff(num_classes=6, use_detr_backbone=False)
        model1 = model1.to(device)
        print("  [PASS] Option 1 model created successfully")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward pass with dummy data."""
    print("\nTesting forward pass...")
    device = get_device()
    batch_size = 2
    
    # Create dummy images
    dummy_img1 = torch.randn(batch_size, 3, 800, 800).to(device)
    dummy_img2 = torch.randn(batch_size, 3, 800, 800).to(device)
    
    try:
        # Test Option 2
        print("  Testing Option 2 forward pass...")
        model2 = DETRPixelDiff(num_classes=6).to(device)
        model2.eval()
        with torch.no_grad():
            outputs2 = model2(dummy_img1, dummy_img2)
            assert hasattr(outputs2, 'logits') or 'logits' in outputs2
            print("  [PASS] Option 2 forward pass successful")
        
        # Test Option 1
        print("  Testing Option 1 forward pass...")
        model1 = DETRFeatureDiff(num_classes=6, use_detr_backbone=False).to(device)
        model1.eval()
        with torch.no_grad():
            outputs1 = model1(dummy_img1, dummy_img2)
            assert hasattr(outputs1, 'logits') or 'logits' in outputs1
            print("  [PASS] Option 1 forward pass successful")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation():
    """Test loss computation."""
    print("\nTesting loss computation...")
    device = get_device()
    batch_size = 2
    
    # Create dummy data
    dummy_img1 = torch.randn(batch_size, 3, 800, 800).to(device)
    dummy_img2 = torch.randn(batch_size, 3, 800, 800).to(device)
    
    # Create dummy targets
    targets = [
        {
            'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32).to(device),
            'class_labels': torch.tensor([1], dtype=torch.long).to(device)
        },
        {
            'boxes': torch.tensor([[0.3, 0.3, 0.1, 0.1]], dtype=torch.float32).to(device),
            'class_labels': torch.tensor([2], dtype=torch.long).to(device)
        }
    ]
    
    try:
        # Test Option 2 with labels
        print("  Testing Option 2 loss computation...")
        model2 = DETRPixelDiff(num_classes=6).to(device)
        model2.train()
        outputs2 = model2(dummy_img1, dummy_img2, labels=targets)
        assert hasattr(outputs2, 'loss') and outputs2.loss is not None
        print(f"  [PASS] Option 2 loss computation successful (loss: {outputs2.loss.item():.4f})")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loader():
    """Test data loading (if data exists)."""
    print("\nTesting data loader...")
    
    # Check if matched annotations exist
    matched_dir = 'data/matched_annotations'
    if not os.path.exists(matched_dir):
        print("  [SKIP] Matched annotations directory not found - skipping data loader test")
        print("         Run: python data_ground_truth_labeller.py")
        return True  # Not a failure, just missing data
    
    # Check if index file exists
    if not os.path.exists('index.txt'):
        print("  [SKIP] index.txt not found - skipping data loader test")
        return True
    
    try:
        # Try to create dataset (will fail if no valid pairs)
        dataset = MovedObjectDataset(
            index_file='index.txt',
            matched_annotations_dir=matched_dir,
            data_dir='.',
            image_size=800
        )
        
        if len(dataset) == 0:
            print("  [SKIP] No valid data pairs found - run data_ground_truth_labeller.py first")
            return True
        
        print(f"  [PASS] Dataset created successfully ({len(dataset)} samples)")
        
        # Test loading a sample
        sample = dataset[0]
        assert 'image1' in sample
        assert 'image2' in sample
        assert 'boxes' in sample
        assert 'labels' in sample
        print("  [PASS] Sample loading successful")
        
        # Test data loader
        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
        batch = next(iter(loader))
        assert 'images1' in batch
        assert 'images2' in batch
        print("  [PASS] DataLoader works correctly")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Data loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("BASIC FUNCTIONALITY TESTS")
    print("="*60)
    
    set_seed(42)
    
    tests = [
        ("Imports", test_imports),
        ("Model Instantiation", test_model_instantiation),
        ("Forward Pass", test_forward_pass),
        ("Loss Computation", test_loss_computation),
        ("Data Loader", test_data_loader),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[FAIL] {test_name} crashed: {e}")
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
        status = "[PASS]" if result else "[FAIL]"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! Implementation is ready for training.")
    else:
        print("\n[WARNING] Some tests failed. Please fix issues before training.")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

