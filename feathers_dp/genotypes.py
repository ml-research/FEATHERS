from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

TABULAR_PRIMITIVES = [
  'none',
  'relu_ln_1',
  'sigmoid_ln_1',
  'tanh_ln_1',
  'relu_ln_2_reddim',
  'relu_ln_2_expdim',
  'sigmoid_2_reddim',
  'sigmoid_2_expdim',
  'tanh_ln_2_reddim',
  'tahn_ln_2_expdim',
  'skip_connect'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
# GENOTYPE = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))
GENOTYPE = Genotype(normal=[('sigmoid_2_expdim', 1), ('sigmoid_2_reddim', 0), ('sigmoid_2_expdim', 2), ('relu_ln_1', 1), ('sigmoid_ln_1', 2), ('sigmoid_ln_1', 1), ('sigmoid_ln_1', 3), ('skip_connect', 4), ('skip_connect', 5), ('sigmoid_ln_1', 2), ('skip_connect', 6), ('sigmoid_ln_1', 5), ('sigmoid_2_expdim', 1), ('sigmoid_ln_1', 7)], normal_concat=range(9, 9), reduce=[('relu_ln_1', 1), ('tanh_ln_1', 0), ('sigmoid_2_reddim', 0), ('tanh_ln_2_reddim', 2), ('tahn_ln_2_expdim', 1), ('tanh_ln_2_reddim', 3), ('relu_ln_2_reddim', 0), ('sigmoid_ln_1', 4), ('tanh_ln_2_reddim', 0), ('tanh_ln_2_reddim', 5), ('sigmoid_2_expdim', 6), ('relu_ln_1', 1), ('skip_connect', 6), ('skip_connect', 3)], reduce_concat=range(9, 9))

