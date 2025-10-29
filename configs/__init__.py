# proj_g    : 2.9K
# proj_f    : 147K
# ln_f      : 384
# split_head: 74.3K
# dense_head: 76.8K
# 1.5M
base_t_96 = dict(
    input_dim=14, embedding_dim=96, num_layers=12, num_heads=4, dropout=0.1
)
# [block]: 5.3M
base_s_192 = dict(
    embedding_dim=192, num_layers=12, num_heads=6, dropout=0.1
)
# 22.2M
base_s_384 = dict(
    input_dim=14, embedding_dim=384, num_layers=12, num_heads=8, dropout=0.2
)
# 43.5M
base_m_384 = dict(
    input_dim=14, embedding_dim=384, num_layers=24, num_heads=8, dropout=0.2
)
# [block]: 85.1M
base_s_768 = dict(
    input_dim=14, embedding_dim=768, num_layers=12, num_heads=12, dropout=0.3
)
# 173M
base_m_768 = dict(
    input_dim=14, embedding_dim=768, num_layers=24, num_heads=16, dropout=0.3
)
# 258M
base_l_768 = dict(
    input_dim=14, embedding_dim=768, num_layers=36, num_heads=16, dropout=0.3
)
# 690M
base_m_1536 = dict(
    input_dim=14, embedding_dim=1536, num_layers=24, num_heads=32, dropout=0.3
)
# 1B
base_l_1536 = dict(
    input_dim=14, embedding_dim=1536, num_layers=36, num_heads=32, dropout=0.3
)
