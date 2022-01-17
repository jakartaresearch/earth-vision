# Experiment Fine-tuning model on RESISC45 Dataset

| Model Architecture | Batch size | Epoch | Optimizer | Learning rate | Train Loss | Train Acc | Val Loss | Val Acc | Test Loss | Test Acc |
| ------------------ | ---------- | ----- | --------- | ------------- | ---------- | --------- | -------- | ------- | ------------ | -------- |
| ResNet50 | 512 | 5 | Adam | 1e-3 | 0.591 | 0.842 | 0.619 | 0.826 | 0.643 | 0.821 |
| ResNet50 | 512 | 15 | AdamW | 1e-3 | 0.363 | 0.896 | 0.486 | 0.846 | 0.493 | 0.846 |
| RegNet y 400mf | 512 | 5 | Adam | 1e-3 | 0.841 | 0.791 | 0.845 | 0.786 | 0.871 | 0.781 | 
| RegNet y 400mf | 512 | 15 | AdamW | 1e-3 | 0.502 | 0.862 | 0.604 | 0.822 | 0.631 | 0.817 | 