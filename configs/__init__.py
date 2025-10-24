# embedding_dim // num_heads %  6 == 0
base_s_192 = dict(
    input_dim=14, embedding_dim=192, num_layers=12, num_heads=3, dropout=0.1
)

base_s_432 = dict(
    input_dim=14, embedding_dim=384, num_layers=12, num_heads=8, dropout=0.2
)
# 88.1M
base_s_768 = dict(
    input_dim=14, embedding_dim=768, num_layers=12, num_heads=16, dropout=0.3
)

base_m_384 = dict(
    input_dim=14, embedding_dim=384, num_layers=24, num_heads=6, dropout=0.1
)

base_m_768 = dict(
    input_dim=14, embedding_dim=768, num_layers=24, num_heads=12, dropout=0.2
)

base_m_1536 = dict(
    input_dim=14, embedding_dim=1536, num_layers=24, num_heads=24, dropout=0.3
)

base_l_768 = dict(
    input_dim=14, embedding_dim=768, num_layers=36, num_heads=12, dropout=0.1
)

base_l_1536 = dict(
    input_dim=14, embedding_dim=1536, num_layers=36, num_heads=24, dropout=0.2
)

base_l_3072 = dict(
    input_dim=14, embedding_dim=3072, num_layers=36, num_heads=48, dropout=0.3
)
