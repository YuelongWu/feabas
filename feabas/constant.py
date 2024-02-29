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
MESH_TRIFINDER_INNERMOST = 2  # the triangle furthest away from boudaries
TRIFINDER_MODE_LIST = ('WHATEVER', 'LEAST_DEFORM', 'INNERMOST')

RENDER_LOCAL_RIGID = 0
RENDER_LOCAL_AFFINE = 1
RENDER_CONTIGEOUS = 2
RENDER_FULL = 3

BLEND_NONE = 0      # no blending, randomly select a tile
BLEND_MAX = 1       # keep the pixel with the largest weight
BLEND_LINEAR = 2    # linear blend the pixels based on their weight

ANNEAL_GLOBAL_RIGID = 0
ANNEAL_GLOBAL_AFFINE = 1
ANNEAL_CONNECTED_RIGID = 2
ANNEAL_CONNECTED_AFFINE = 3
ANNEAL_COPY_EXACT = 4

# material model type
MATERIAL_MODEL_ENG = 0    # Engineering strain & stress
MATERIAL_MODEL_SVK = 1    # St. Venant-Kirchhoff model
MATERIAL_MODEL_NHK = 2    # Neo-Hookean model
MATERIAL_MODEL_LIST =('MATERIAL_MODEL_ENG', 'MATERIAL_MODEL_SVK', 'MATERIAL_MODEL_NHK')

FFT_CONF_NONE = 0       # confidence values set to constant 1
FFT_CONF_STD = 1        # confidence value derived from the standard deviation
FFT_CONF_MIRROR = 2     # confidence value derived from the convolution

DEFAULT_RESOLUTION = 4.0
DEFAULT_THICKNESS = 30.0
EPSILON0 = 1e-5
