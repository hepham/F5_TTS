@echo off
echo Installing PyTorch with CUDA 12.4 support...
echo This will enable GPU acceleration for F5-TTS

pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

echo.
echo PyTorch installation completed!
echo Testing CUDA availability...

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo Press any key to continue...
pause 