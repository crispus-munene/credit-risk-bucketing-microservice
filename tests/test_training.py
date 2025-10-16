import subprocess
import sys
def test_train_runs():
    """Run train_model.py and ensure it doesn't crash"""
    result= subprocess.run(
        [sys.executable,"src/models/train_model.py"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f'Training failed: {result.stderr}'