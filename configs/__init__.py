# embedding 82K
# - embedding: (256*5+1)*vocal_dim
#   for default config, vocal_dim=64, embedding has 82.0K params
# - proj_x, proj_o, ...: vocal_dim*14*embedding_dim

# +--------+----+
# | model  | hd |
# +--------+----+
# | tiny   | 16 |
# | small  | 32 |
# | middle | 64 |
# +--------+----+

# +--------------------+
# | base_t_96  |  448K |
# +--------------------+
base_t_96 = dict(
    embedding_dim=96,
    num_layers=6,
    num_heads=6,
    dropout=0.1
)
# +--------------------+
# | base_s_96  |  897K |
# | base_s_192 |  3.6M |
# | base_s_384 | 14.2M |
# | base_s_768 | 56.7M |
# +--------------------+
base_s_96 = dict(
    embedding_dim=96,
    num_layers=12,
    num_heads=3,
    dropout=0.1
)
base_s_192 = dict(
    embedding_dim=192,
    num_layers=12,
    num_heads=6,
    dropout=0.1
)
base_s_384 = dict(
    embedding_dim=384,
    num_layers=12,
    num_heads=12,
    dropout=0.1
)
# block: 56.7 M
base_s_768 = dict(
    embedding_dim=768,
    num_layers=12,
    num_heads=24,
    dropout=0.1
)
# +--------------------+
# | base_m_192 |  7.1M |
# | base_m_384 | 28.4M |
# | base_m_768 |  113M |
# +--------------------+
base_m_192 = dict(
    embedding_dim=192,
    num_layers=24,
    num_heads=3,
    dropout=0.2
)
# block: 28.4 M
base_m_384 = dict(
    embedding_dim=384,
    num_layers=24,
    num_heads=6,
    dropout=0.2
)
base_m_768 = dict(
    embedding_dim=768,
    num_layers=24,
    num_heads=12,
    dropout=0.2
)

__all__ = ['base_t_96', 'base_s_96', 'base_s_192', 'base_s_384', 'base_s_768', 'base_m_192', 'base_m_384', 'base_m_768']
