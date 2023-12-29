from collections import defaultdict, OrderedDict

import json
import numpy as np
from scipy.interpolate import interp1d

from feabas import common
import feabas.constant as const


DTYPE = np.float32  # single-precision

class Material:
    """
    Class to represent a material with a predefined material model &  mechanical
    properties.
    Kwargs:
        enable_mesh(bool): if enable_mesh set to False, the material is treated
            as excluded region and no mesh will be generated on it.
        area_constraint(float): maximum triangle area constraint multiplier used
            to feed into triangulation function for meshing. The actual area
            constraint would be this value multiplied by the size settings
            during meshing. If set to 0, no constraint applied.
        render(bool): whether to render.
        render_weight(float): when mesh collides, materials with larger render
            weight have higher priority when render.
        type(int): material type. Possible options include 0::engineering,
            1::St.Venant-Kirchhoff, 2::Neo-Hookean.
        stiffness_multiplier(float): constant multiplied to the stiffness
            function.
        stiffness_func_factory(str)/stiffness_func_params(dict): function
            factory that produces function that maps strain-stress relationship.
            The function takes in the area change ratios, and returns the
            stiffness modifier. Used for special material (e.g material can
            freely expand but harder to compress).
        poisson_ratio(float): Poisson ratio. For Neo-Hookean, only implemeted 0
            Poisson's ratio.
        id(int): id number of the material used to feed into region props in the
            triagulation function. If set to None, will use a class counter.
            Otherwise, the user is responsible for this to be unique.
        mask_label(int or RGB triplet): designated label in the mask image.
    """
    uid = 1
    def __init__(self, **kwargs):
        self.enable_mesh = kwargs.get('enable_mesh', True)
        self.area_constraint = kwargs.get('area_constraint', 1.0)
        self.render = kwargs.get('render', True)
        self.render_weight = kwargs.get('render_weight', 1.0)
        mat_type = kwargs.get('type', const.MATERIAL_MODEL_ENG)
        if isinstance(mat_type, str):
            mat_type = const.MATERIAL_MODEL_LIST.index(mat_type.upper())
        self._type = mat_type
        self._stiffness_multiplier = kwargs.get('stiffness_multiplier', 1.0)
        self._poisson_ratio = kwargs.get('poisson_ratio', 0.0)
        self.mask_label = kwargs.get('mask_label', None)
        stiffness_func_factory = kwargs.get('stiffness_func_factory', None)
        stiffness_func_params = kwargs.get('stiffness_func_params', {}).copy()
        self.update_stiffness_func(stiffness_func_factory, stiffness_func_params)
        uid = kwargs.get('uid', None)
        if uid is None:
            self.uid = Material.uid
            Material.uid += 1
        else:
            self.uid = uid


    def to_dict(self):
        out = {
            'enable_mesh': self.enable_mesh,
            'area_constraint': self.area_constraint,
            'render': self.render,
            'render_weight': self.render_weight,
            'type': const.MATERIAL_MODEL_LIST[self._type],
            'stiffness_multiplier': self._stiffness_multiplier,
            'poisson_ratio': self._poisson_ratio,
            'uid': self.uid
            }
        if self.mask_label is not None:
            out['mask_label'] = self.mask_label
        if callable(self._stiffness_func_factory):
            if self._stiffness_func_factory.__name__ == '<lambda>':
                raise TypeError
            else:
                func_mod = self._stiffness_func_factory.__module__
                func_name = self._stiffness_func_factory.__name__
                out['stiffness_func_factory'] = func_mod + '.' + func_name
        elif isinstance(self._stiffness_func_factory, (str, type(None))):
            out['stiffness_func_factory'] = self._stiffness_func_factory
        else:
            raise TypeError
        stiffness_func_params = {}
        for key, val in self._stiffness_func_params.items():
            if isinstance(val, np.ndarray):
                val = val.tolist()
            stiffness_func_params[key] = val
        out['stiffness_func_params'] = stiffness_func_params
        return out


    def update_stiffness_func(self, stiffness_func_factory=None, stiffness_func_params=None):
        self._stiffness_func_factory = stiffness_func_factory
        if stiffness_func_params is None:
            stiffness_func_params = {}
        self._stiffness_func_params = stiffness_func_params
        if self._stiffness_func_factory is None:
            self._stiffness_func = None
        elif isinstance(self._stiffness_func_factory, str):
            if 'lambda' in self._stiffness_func_factory:
                self._stiffness_func = eval(self._stiffness_func_factory)
            else:
                stiffness_func_factory = common.load_plugin(self._stiffness_func_factory)
                self._stiffness_func = stiffness_func_factory(**self._stiffness_func_params)
        elif callable(self._stiffness_func_factory):
            self._stiffness_func = self._stiffness_func_factory(**self._stiffness_func_params)
        else:
            raise TypeError


    def shape_matrix_from_vertices(self, tripts):
        """
        generate element shape matrices used to construct stiffness matrix.
        Args:
            tripts(Nx3x2 np.ndarray): xy coordinates of the triangle vertices.
        Return:
            shape_matrix(Nx4x6 np.ndarray): shape matrix for interpolation.
            triangle_areas(Nx1x1 np.ndarray): area of the mesh triangles.
        """
        # stiffness matrix: [u1, v1, u2, v2, u3, v3]
        tripts = tripts.reshape(-1, 3, 2)
        trinum = tripts.shape[0]
        tripts_m = tripts.mean(axis=1, keepdims=True)
        tripts = (tripts - tripts_m).astype(DTYPE)
        tripts_pad = np.pad(tripts, ((0,0),(0,0),(0,1)), mode='constant', constant_values=1.0)
        J = np.linalg.inv(tripts_pad)[:,:2,:]
        B = np.zeros((trinum, 4, 6), dtype=DTYPE)
        B[:,0,0::2] = J[:,0,:]
        B[:,1,0::2] = J[:,1,:]
        B[:,2,1::2] = J[:,0,:]
        B[:,3,1::2] = J[:,1,:]
        # caculate areas
        v0 = (tripts[:,1,:] - tripts[:,0,:])/100
        v1 = (tripts[:,2,:] - tripts[:,1,:])/100
        areas = np.absolute(np.cross(v0, v1)).reshape(-1,1,1)
        return B, areas


    def element_stiffness_matrices_from_shape_matrices(self, Ms, uv=None, **kwargs):
        """
        given the shape matrices, generate the element stiffness matrix.
        Args:
            Ms(np.ndarrays): shape matrices generated by method
                shape_matrix_from_vertices. B(Nx4x6), area(Nx1x1).
            uv(Nx6): xy displacement. Each element follows the order
                u1, v1, u2, v2, u3, v3.
            check_flip(bool): check if any triangles are flipped.
            continue_on_flip(bool): whether to continue with flipped triangles
                detected.
        return:
            element_stiffness(Nx6x6): element tangent stiffness matrix used at
                stiffness matrix assembly, computed at the current displacement.
            stress(Nx6x1): stress computed at the current displacement. Used in
                Newton-Raphson iterations.
            flipped(bool): if any triangle is flipped.
        """
        check_flip = kwargs.get('check_flip', None)
        continue_on_flip = kwargs.get('continue_on_flip', False)
        B, areas = Ms
        flipped = None
        if (self._stiffness_multiplier == 0) or (B is None):
            return None, None, flipped
        trinum = B.shape[0]
        if uv is None:
            uv = np.zeros((trinum, 6, 1), dtype=DTYPE, order='C')
        else:
            uv = uv.astype(DTYPE).reshape(-1, 6, 1)
        if check_flip is None:
            if self._type == const.MATERIAL_MODEL_ENG:
                check_flip = False
            else:
                check_flip = True
        if self._type == const.MATERIAL_MODEL_ENG:
            # Engineering strain & stress
            if (self._stiffness_func is not None) or check_flip:
                Ft = (B @ uv).reshape(-1,2,2) + np.eye(2, dtype=DTYPE)
                J = np.linalg.det(Ft).reshape(-1,1,1)
                if check_flip:
                    flipped = np.any(J <= 0, axis=None)
                    if (not continue_on_flip) and flipped:
                        return None, None, flipped
            Bn = np.array([[1,0,0,0],[0,0,0,1],[0,1,1,0]], dtype=DTYPE) @ B
            D = np.eye(3, dtype=DTYPE)
            D[[0,1],[1,0]] = self._poisson_ratio
            D[-1,-1] = (1 - self._poisson_ratio) / 2
            K = np.swapaxes(Bn, 1, 2) @ D @ Bn
            K = areas * K
            P = K @ uv
        elif self._type == const.MATERIAL_MODEL_SVK:
            # St. Venant-Kirchhoff
            Ft = (B @ uv).reshape(-1,2,2) + np.eye(2, dtype=DTYPE)
            if (self._stiffness_func is not None) or check_flip:
                J = np.linalg.det(Ft).reshape(-1,1,1)
                if check_flip:
                    flipped = np.any(J <= 0, axis=None)
                    if (not continue_on_flip) and flipped:
                        return None, None, flipped
            FtT = np.swapaxes(Ft,1,2)
            Et = 0.5*(FtT@Ft - np.eye(2, dtype=DTYPE))
            E = np.array([[1,0,0,0],[0,0,0,1],[0,1,1,0]], dtype=DTYPE) @ Et.reshape(-1, 4, 1)
            Bc = np.array([[1,0,1,0],[0,1,0,1]], dtype=DTYPE) @ B
            Fc = np.tile(FtT, (1,1,3))
            Bn_top = Bc * Fc
            Bn_bot = np.sum(Bc * Fc[:,::-1,:], axis=1, keepdims=True)
            Bn = np.concatenate((Bn_top, Bn_bot), axis=1) # Nx3x6
            D = np.eye(3, dtype=DTYPE)
            D[[0,1],[1,0]] = self._poisson_ratio
            D[-1,-1] = (1 - self._poisson_ratio) / 2
            S = D @ E # Nx3x1
            Sg  = np.zeros((trinum,4,4), dtype=DTYPE)
            Sg[:,0,0] = S[:,0,0]
            Sg[:,2,2] = S[:,0,0]
            Sg[:,1,1] = S[:,1,0]
            Sg[:,3,3] = S[:,1,0]
            Sg[:,0,1] = S[:,2,0]
            Sg[:,1,0] = S[:,2,0]
            Sg[:,2,3] = S[:,2,0]
            Sg[:,3,2] = S[:,2,0]
            P = np.swapaxes(Bn, 1, 2) @ S
            K = np.swapaxes(Bn, 1, 2) @ D @ Bn + np.swapaxes(B, 1, 2) @ Sg @ B
            P = areas * P
            K = areas * K
        elif self._type == const.MATERIAL_MODEL_NHK:
            # Neo-Hookean
            Ft = (B @ uv).reshape(-1,2,2) + np.eye(2, dtype=DTYPE)
            J = np.linalg.det(Ft).reshape(-1,1,1)
            if check_flip:
                flipped = np.any(J <= 0, axis=None)
                if (not continue_on_flip) and flipped:
                    return None, None, flipped
            U = np.array([[0,0,0,1],[0,0,-1,0],[0,-1,0,0],[1,0,0,0]],dtype=DTYPE)
            F = Ft.reshape(-1,4,1)
            Fu = U @ F
            P = np.swapaxes(B, 1, 2) @ (np.eye(4, dtype=DTYPE) - U/J) @ F
            K = np.swapaxes(B, 1, 2) @ (np.eye(4, dtype=DTYPE) - U/J + (Fu@np.swapaxes(Fu,1,2))/(J**2)) @ B
            P = 0.5 * areas * P
            K = 0.5 * areas * K
        else:
            raise NotImplementedError
        if self._stiffness_func is not None:
            modifier = self._stiffness_func(J)
            K = K * modifier.reshape(-1,1,1)
            P = P * modifier.reshape(-1,1,1)
        if self._stiffness_multiplier != 1:
            K = self._stiffness_multiplier * K
            P = self._stiffness_multiplier * P
        return K, P, flipped


    def element_stiffness_matrices_from_vertices(self, tripts, uv=None, check_flip=None):
        """
        combination of method self.shape_matrix_from_vertices and method
        self.element_stiffness_matrices_from_shape_matrices
        """
        Ms = self.shape_matrix_from_vertices(tripts)
        return self.element_stiffness_matrices_from_shape_matrices(Ms, uv=uv, check_flip=check_flip)


    @property
    def stiffness_multiplier(self):
        return self._stiffness_multiplier


    @property
    def is_linear(self):
        return (self._type == const.MATERIAL_MODEL_ENG) and (self._stiffness_func is None)



MATERIAL_EXCLUDE = Material(enable_mesh=False,
                         uid=0,
                         mask_label=255,
                         stiffness_multiplier=0.0,
                         render=False,
                         render_weight=1.0e-6)

MATERIAL_DEFAULT = Material(enable_mesh=True,
                            area_constraint=1,
                            type=const.MATERIAL_MODEL_ENG,
                            stiffness_multiplier=1.0,
                            poisson_ratio=0.0,
                            uid=-1,
                            mask_label=0)



class MaterialTable:
    """
    A collection of materials.
    Kwargs:
        table (dict): dictionary storing material label - material pairs.
        default_material (feabas.material.Material): the default material to use
            when querying a nonexistent material label. If set to None, and
            'default' label in the table, use that as the default material.
            Otherwise, use MATERIAL_DEFAULT as the default material.
    """
    def __init__(self, table=None, default_material=None):
        if table is None:
            table = {}
        if default_material is not None:
            table['default'] = default_material
            default_factory = lambda: default_material
        elif 'default' in table:
            default_factory = lambda: table['default']
        else:
            table['default'] = MATERIAL_DEFAULT
            default_factory = lambda: MATERIAL_DEFAULT
        self._table = defaultdict(default_factory)
        if 'exclude' not in table:
            table['exclude'] =  MATERIAL_EXCLUDE
        self._table.update(table)


    @classmethod
    def from_json(cls, jsonname, stream=False, default_material=None):
        if stream:
            dct = json.loads(jsonname)
        else:
            with open(jsonname, 'r') as f:
                dct = json.load(f)
        table = {}
        for lbl, props in dct.items():
            material = Material(**props)
            table[lbl] = material
        return cls(table=table, default_material=default_material)


    def copy(self):
        table = self._table.copy()
        return self.__class__(table=table)


    def save_to_json(self, jsonname=None):
        outdict = {}
        for lbl, material in self._table.items():
            outdict[lbl] = material.to_dict()
        json_str = json.dumps(outdict, indent = 2)
        if jsonname is not None:
            with open(jsonname, 'w') as f:
                f.write(json_str)
        return json_str


    def __getitem__(self, key):
        return self._table[key]


    def __iter__(self):
        for labelname, mat in self._table.items():
            yield labelname, mat


    def add_material(self, label, material, force_update=True):
        if force_update or (label not in self._table):
            self._table[label] = material


    def combine_material_table(self, mtb, force_update=False):
        if force_update:
            self._table.update(mtb.named_table)
        else:
            for lbl, m in mtb:
                if lbl not in self._table:
                    self._table[lbl] = m


    @property
    def name_to_label_mapping(self):
        mapper = OrderedDict()
        for name, mat in self._table.items():
            lbl = mat.mask_label
            if lbl is not None:
                mapper[name] = lbl
        return mapper


    @property
    def named_table(self):
        return self._table


    @property
    def id_table(self):
        if 'default' in self._table:
            default_factory = lambda: self._table['default']
        else:
            default_factory = None
        id_tabel = defaultdict(default_factory)
        for val in self._table.values():
            id_tabel[val.uid] = val
        return id_tabel



# stiffness function factories.
# strain = 1: no change; strain = 0: flip.
def asymmetrical_elasticity(**params):
    strain = params.get('strain', [0, 0.75, 1, 1.01])
    stiffness = params.get('stiffness', [1.5, 1, 0.5, 0])
    f = interp1d(strain, stiffness, kind='linear', bounds_error=False, fill_value=(stiffness[0], stiffness[-1]))
    return f
