SPATIAL_SIMPLIFY_REGION = 0
SPATIAL_SIMPLIFY_GROUP = 1
SPATIAL_SIMPLIFY_SEGMENT = 2

MESH_GEAR_INITIAL = -1  # initial fixed vertices
MESH_GEAR_FIXED = 0     # fixed vertices
MESH_GEAR_MOVING = 1    # moving vertices
MESH_GEAR_STAGING = 2   # moving vertices before validity checking and committing

MESH_HISTORY_LEN = 5    # length of history of mesh operations to keep (used for generate cache keys)

# material model type
MATERIAL_MODEL_ENG = 0    # Engineering strain & stress
MATERIAL_MODEL_SVK = 1    # St. Venant-Kirchhoff model
MATERIAL_MODEL_NHK = 2    # Neo-Hookean model

EPSILON0 = 1e-5
