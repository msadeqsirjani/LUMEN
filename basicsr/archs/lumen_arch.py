from basicsr.archs.lumen.model import LUMEN
from basicsr.utils.registry import ARCH_REGISTRY

# Register all three LUMEN variants so they can be instantiated from YAML:
#   network_g:
#     type: LUMEN
#     embed_dim: 32
#     depths: [4, 4]
#     ...
ARCH_REGISTRY.register()(LUMEN)
