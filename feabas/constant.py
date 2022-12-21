SPATIAL_SIMPLIFY_REGION = 0
SPATIAL_SIMPLIFY_GROUP = 1
SPATIAL_SIMPLIFY_SEGMENT = 2

MESH_GEAR_INITIAL = -1  # initial fixed vertices
MESH_GEAR_FIXED = 0     # fixed vertices
MESH_GEAR_MOVING = 1    # moving vertices
MESH_GEAR_STAGING = 2   # moving vertices before validity checking and committing
MESH_GEARS = (MESH_GEAR_INITIAL, MESH_GEAR_FIXED, MESH_GEAR_MOVING, MESH_GEAR_STAGING)

# tri_finder policy upon triangle overlaps
MESH_TRIFINDER_WHATEVER = 0         # whichever triangle
MESH_TRIFINDER_LEAST_DEFORM = 1     # the least deformed triangle
MESH_TRIFINDER_LEAST_INNERMOST = 2  # the triangle furthest away from boudaries

# material model type
MATERIAL_MODEL_ENG = 0    # Engineering strain & stress
MATERIAL_MODEL_SVK = 1    # St. Venant-Kirchhoff model
MATERIAL_MODEL_NHK = 2    # Neo-Hookean model

EPSILON0 = 1e-5
