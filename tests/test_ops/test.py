import torch
from test_fused_bias_leakyrelu import TestFusedBiasLeakyReLU

a = TestFusedBiasLeakyReLU()
a.setup_class()
a.test_gradient()
print('done')
