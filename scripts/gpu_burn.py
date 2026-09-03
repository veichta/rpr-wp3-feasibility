# 60 s of bf16 matmuls on every visible GPU: a GSSR smoke test.
import time, torch
xs = [torch.randn(8192, 8192, device=f"cuda:{i}", dtype=torch.bfloat16) for i in range(torch.cuda.device_count())]
t = time.time()
while time.time() - t < 60:
    for x in xs:
        x @ x
torch.cuda.synchronize()
print("burn ok on", len(xs), "gpus")
