"""
Setup Checker for REAL Privacy Federated Learning
Run this to verify all files are present and working
"""

import os
import sys

print("🔍 Checking REAL Privacy Federated Learning Setup")
print("=" * 50)

# Required files
required_files = [
    'federated_clients.py',
    'real_privacy_utils.py', 
    'real_main_server.py',
    'alzheimer.csv'
]

optional_files = [
    'REAL_PRIVACY_SETUP.md',
    'FAKE_vs_REAL_comparison.py'
]

# Check required files
print("📂 Checking required files:")
missing_files = []
for file in required_files:
    if os.path.exists(file):
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file} - MISSING!")
        missing_files.append(file)

# Check optional files
print("\n📂 Checking optional files:")
for file in optional_files:
    if os.path.exists(file):
        print(f"  ✅ {file}")
    else:
        print(f"  ⚠️  {file} - Optional (recommended)")

# Check Python packages
print("\n📦 Checking required packages:")
required_packages = [
    ('torch', 'PyTorch'),
    ('pandas', 'Data handling'),
    ('numpy', 'Numerical computing'),
    ('sklearn', 'Machine learning utilities'),
    ('matplotlib', 'Plotting'),
    ('tenseal', 'TRUE Homomorphic Encryption')
]

missing_packages = []
for package, description in required_packages:
    try:
        __import__(package)
        print(f"  ✅ {package} - {description}")
    except ImportError:
        print(f"  ❌ {package} - MISSING! ({description})")
        missing_packages.append(package)

# Summary
print("\n" + "=" * 50)
print("📊 SETUP SUMMARY")
print("=" * 50)

if not missing_files and not missing_packages:
    print("🎉 ALL CHECKS PASSED!")
    print("✅ Ready to run TRUE privacy-preserving federated learning")
    print("\n🚀 Next steps:")
    print("   1. Run: python real_main_server.py")
    print("   2. Watch TRUE privacy protection in action!")
    print("   3. Enjoy server blindness to individual gradients! 🛡️")
    
elif missing_files:
    print("❌ MISSING FILES:")
    for file in missing_files:
        print(f"   - {file}")
    print("\n📝 Action needed:")
    print("   Copy the missing files to this directory")
    
elif missing_packages:
    print("❌ MISSING PACKAGES:")
    for package in missing_packages:
        print(f"   - {package}")
    print("\n📦 Install missing packages:")
    print("   pip install " + " ".join(missing_packages))
    print("\n⚠️  Special note for TenSEAL:")
    print("   If 'pip install tenseal' fails, try:")
    print("   conda install -c conda-forge tenseal")

# Test basic imports if everything is available
if not missing_files and not missing_packages:
    print("\n🧪 Testing basic functionality...")
    try:
        from federated_clients import ClientManager, AlzheimerNet
        print("  ✅ Federated clients import successful")
        
        from real_privacy_utils import TrueSecureAggregator
        print("  ✅ REAL privacy utils import successful")
        
        import tenseal as ts
        print("  ✅ TenSEAL (TRUE HE) import successful")
        
        print("\n🛡️  TRUE Privacy System Ready!")
        
    except Exception as e:
        print(f"  ❌ Import test failed: {e}")
        print("  Check that all files are properly saved")

print("\n📍 Current directory contents:")
files = [f for f in os.listdir('.') if f.endswith(('.py', '.csv', '.md'))]
for file in sorted(files):
    size = os.path.getsize(file) / 1024  # KB
    print(f"  {file} ({size:.1f} KB)")
