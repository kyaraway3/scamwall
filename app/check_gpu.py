import torch

print("--- GPU確認結果 ---")
available = torch.cuda.is_available()
print(f"GPUが使えるか: {available}")

if available:
    print(f"GPUの名前: {torch.cuda.get_device_name(0)}")
    print(f"CUDAバージョン: {torch.version.cuda}")
else:
    print("残念ながら現在は CPU モードです。")