#!/usr/bin/env python3
"""
MediScan Disease Classifier — Startup Script
Auto-installs all required packages if missing, then starts the server.
"""
import sys
import os
import subprocess

REQUIRED = {
    'flask': 'flask',
    'sklearn': 'scikit-learn',
    'numpy': 'numpy',
    'scipy': 'scipy',
    'joblib': 'joblib',
    'reportlab': 'reportlab',
}

def check_and_install():
    missing = []
    for module, package in REQUIRED.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"\n📦 Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--quiet', *missing
        ])
        print("✅ All packages installed!\n")
    else:
        print("✅ All dependencies already satisfied.\n")

def retrain_if_needed():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'classifier.pkl')
    if not os.path.exists(model_path):
        print("🧠 Training ML model (first-time setup)...")
        result = subprocess.run([sys.executable, 'train_model.py'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Training failed:", result.stderr)
            sys.exit(1)
        print("✅ Model trained and saved!\n")

if __name__ == '__main__':
    print("=" * 60)
    print("  [MediScan] — AI Disease Classifier")
    print("=" * 60)

    check_and_install()
    retrain_if_needed()

    import app as app_module
    app_module.init_db()
    app = app_module.app
    print("🌐 Starting server at: http://localhost:5000")
    print("   Press CTRL+C to stop\n")
    app.run(debug=False, port=5000, host='0.0.0.0')
