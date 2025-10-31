base_t_96 = dict(
    embedding_dim=96,
    num_layers=6,
    num_heads=6,
    dropout=0.1
)
# for small, head dimension is 32
# [block]: 5.3M
base_s_192 = dict(
    embedding_dim=192,
    num_layers=12,
    num_heads=6,
    dropout=0.1
)
#
base_s_384 = dict(
    embedding_dim=384,
    num_layers=12,
    num_heads=12,
    dropout=0.1
)
# [block]: 85.1M
base_s_768 = dict(
    embedding_dim=768,
    num_layers=12,
    num_heads=24,
    dropout=0.3
)
# for middle, head dimension is 64
base_m_384 = dict(
    embedding_dim=384,
    num_layers=24,
    num_heads=6,
    dropout=0.2
)
# [block]: 85.1M
base_m_768 = dict(
    embedding_dim=768,
    num_layers=12,
    num_heads=12,
    dropout=0.3
)
# 173M
base_m_768 = dict(
    embedding_dim=768,
    num_layers=24,
    num_heads=16,
    dropout=0.3
)
# 258M
base_l_768 = dict(
    embedding_dim=768,
    num_layers=36,
    num_heads=16,
    dropout=0.3
)
# 690M
base_m_1536 = dict(
    embedding_dim=1536,
    num_layers=24,
    num_heads=32,
    dropout=0.3
)
# 1B
base_l_1536 = dict(
    embedding_dim=1536,
    num_layers=36,
    num_heads=32,
    dropout=0.3
)
