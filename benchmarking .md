## salt unet

- times in seconds: 82, 58, 33,
  {
  percentage: 0.25,
  total_params: 1,179,121
  train: 850 samples,
  val: 150 samples,
  batch_size: 32,
  dropout: 0.1,
  epoch: 50,
  early: ~30 - 40,
  early_patience: 10
  }

- times in seconds: 82, 58, 33,
- 5gb gpu ram
  {
  percentage: 1,
  total_params: 1,179,121
  train: 3400 samples,
  val: 150 samples,
  batch_size: 128, # this is 4x larger
  dropout: 0.1,
  epoch: 50,
  early: ~30,
  early_patience: 10
  }
