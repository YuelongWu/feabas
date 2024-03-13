from collections import namedtuple
import cv2
import importlib
import os

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
import scipy.sparse.csgraph as csgraph



Match = namedtuple('Match', ('xy0', 'xy1', 'weight'))


def imread(path, **kwargs):
    flag = kwargs.get('flag', cv2.IMREAD_UNCHANGED)
    if path.startswith('gs://') or path.startswith('s3://') or path.startswith('http://') or path.startswith('https://'):
        if path.lower().endswith('.png'):
            driver = 'png'
        elif path.lower().endswith('.bmp'):
            driver = 'bmp'
        elif path.lower().endswith('.tif') or path.lower().endswith('.tiff'):
            driver = 'tiff'
        elif path.lower().endswith('.jpg') or path.lower().endswith('.jpeg'):
            driver = 'jpeg'
        elif path.lower().endswith('.avif'):
            driver = 'avif'
        elif path.lower().endswith('.webp'):
            driver = 'webp'
        else:
            raise ValueError(f'format not supported: {path}')
        import tensorstore as ts
        js_spec = {'driver': driver, 'kvstore': path}
        try:
            ts_data = ts.open(js_spec).result()
            img = ts_data.read().result()
            if len(img.shape) < 3:
                num_channels = 1
            else:
                num_channels = img.shape[-1]
            if flag == cv2.IMREAD_GRAYSCALE and num_channels != 1:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif flag == cv2.IMREAD_COLOR and num_channels == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        except ValueError:
            img = None
    else:
        if path.startswith('file://'):
            path = path.replace('file://', '')
        img = cv2.imread(path, flag)
    return img


def imwrite(path, image):
    if path.startswith('gs://'):
        from google.cloud import storage
        plist = path.replace('gs://', '').split('/')
        plist = [s for s in plist if s]
        bucket = plist[0]
        relpath = '/'.join(plist[1:])
        _, encoded_img = cv2.imencode('.PNG', image)
        client = storage.Client()
        bucket = client.get_bucket(bucket)
        blob = bucket.blob(relpath)
        blob.upload_from_string(encoded_img.tobytes(), content_type='image/png')
    else:
        if path.startswith('file://'):
            path = path.replace('file://', '')
        return cv2.imwrite(path, image)


def inverse_image(img, dtype=np.uint8):
    if dtype is None:
        if isinstance(img, np.ndarray):
            dtype = img.dtype
        else:
            dtype = type(img)
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        intmx = np.iinfo(dtype).max
        return intmx - img
    elif np.issubdtype(dtype, np.floating):
        return -img
    else:
        raise TypeError(f'{dtype} not invertable.')


def z_order(indices, base=2):
    """
    generating z-order from multi-dimensional indices.
    Args:
        indices(Nxd ndarray): indexing arrays with each colume as a dimension.
            Integer entries assumed.
    Return:
        z-order(Nx1 ndarray): the indices that would sort the points in z-order.
    """
    ndim = indices.shape[-1]
    indices = indices - indices.min(axis=0)
    indices_casted = np.zeros_like(indices)
    pw = 0
    while np.any(indices > 0):
        mod = indices % base
        indices_casted = indices_casted + mod * (base ** (ndim * pw))
        indices = np.floor(indices / base)
        pw += 1
    z_order_score = np.sum(indices_casted * (base ** np.arange(ndim)), axis=-1)
    return np.argsort(z_order_score, kind='stable')


def render_by_subregions(map_x, map_y, mask, img_loader, fileid=None,  **kwargs):
    """
    break the render job to small regions in case the target source image is
    too large to fit in RAM.
    """
    rintp = kwargs.get('remap_interp', cv2.INTER_LANCZOS4)
    mx_dis = kwargs.get('mx_dis', 16300)
    fillval = kwargs.get('fillval', img_loader.default_fillval)
    dtype_out = kwargs.get('dtype_out', img_loader.dtype)
    return_empty = kwargs.get('return_empty', False)
    seeds = kwargs.get('seeds', None)
    if isinstance(rintp, str):
        rintp_dict = {'NEAREST': cv2.INTER_NEAREST,
                      'LINEAR': cv2.INTER_LINEAR,
                      'CUBIC': cv2.INTER_CUBIC,
                      'LANCZOS': cv2.INTER_LANCZOS4}
        rintp = rintp_dict.get(rintp.upper(), cv2.INTER_LANCZOS4)
    mx_dis = np.atleast_1d(mx_dis)
    if map_x.size == 0:
        return None
    if not np.any(mask, axis=None):
        if return_empty:
            return np.full_like(map_x, fillval, dtype=dtype_out)
        else:
            return None
    tile_ht, tile_wd = mask.shape[:2]
    imgt = np.full_like(map_x, fillval, dtype=dtype_out)
    to_render = mask
    if seeds is None:
        seed_indices = np.empty((0, 2), dtype=np.int8)
    else:
        seeds = np.atleast_1d(seeds).ravel()
        num_r = seeds[0]
        num_c = seeds[-1]
        idx0 = np.unique(np.round(np.arange(tile_ht*0.5/num_r, tile_ht, tile_ht/num_r)))
        idx1 = np.unique(np.round(np.arange(tile_wd*0.5/num_c, tile_wd, tile_wd/num_c)))
        seed_idx0, seed_idx1 = np.meshgrid(idx0.astype(np.int32), idx1.astype(np.int32))
        seed_idx0, seed_idx1 = seed_idx0.ravel(), seed_idx1.ravel()
        seed_indices = np.stack((seed_idx0, seed_idx1), axis=-1)
        sel_idx = to_render[seed_idx0, seed_idx1]
        seed_indices = seed_indices[sel_idx]
    multichannel = False
    while np.any(to_render, axis=None):
        if seed_indices.size > 0:
            indx0_sel = seed_indices[0, 0]
            indx1_sel = seed_indices[0, 1]
        else:
            indx0, indx1 = np.nonzero(to_render)
            indx0_sel = indx0[indx0.size//2]
            indx1_sel = indx1[indx1.size//2]
        xx0 = map_x[indx0_sel, indx1_sel]
        yy0 = map_y[indx0_sel, indx1_sel]
        mskt = (np.abs(map_x - xx0) < mx_dis[-1]) & (np.abs(map_y - yy0) < mx_dis[0]) & to_render
        xmin = np.floor(map_x[mskt].min()) - 4 # Lanczos 8x8 kernel
        xmax = np.ceil(map_x[mskt].max()) + 4
        ymin = np.floor(map_y[mskt].min()) - 4
        ymax = np.ceil(map_y[mskt].max()) + 4
        bbox = (int(xmin), int(ymin), int(xmax), int(ymax))
        if fileid is None:
            img0 = img_loader.crop(bbox, **kwargs)
        else:
            img0 = img_loader.crop(bbox, fileid, **kwargs)
        if img0 is None:
            to_render = to_render & (~mskt)
            continue
        if (len(img0.shape) > 2) and (not multichannel):
            # multichannel
            num_channel = img0.shape[-1]
            imgt = np.stack((imgt, )*num_channel, axis=-1)
            multichannel = True
        cover_ratio = np.sum(mskt) / mskt.size
        if cover_ratio > 0.25:
            map_xt = map_x - xmin
            map_yt = map_y - ymin
            imgtt = cv2.remap(img0, map_xt.astype(np.float32), map_yt.astype(np.float32),
                interpolation=rintp, borderMode=cv2.BORDER_CONSTANT, borderValue=fillval)
            if multichannel:
                mskt3 = np.stack((mskt, )*num_channel, axis=-1)
                imgtt = imgtt.reshape(mskt3.shape)
                imgt[mskt3] = imgtt[mskt3]
            else:
                imgt[mskt] = imgtt[mskt]
        else:
            map_xt = map_x[mskt] - xmin
            map_yt = map_y[mskt] - ymin
            N_pad = int(np.ceil((map_xt.size)**0.5))
            map_xt_pad = np.pad(map_xt, (0, N_pad**2 - map_xt.size)).reshape(N_pad, N_pad)
            map_yt_pad = np.pad(map_yt, (0, N_pad**2 - map_yt.size)).reshape(N_pad, N_pad)
            imgt_pad = cv2.remap(img0, map_xt_pad.astype(np.float32), map_yt_pad.astype(np.float32),
                interpolation=rintp, borderMode=cv2.BORDER_CONSTANT, borderValue=fillval)
            if multichannel:
                imgtt = imgt_pad.reshape(-1, num_channel)
                imgtt = imgtt[:(map_xt.size), :]
                mskt3 = np.stack((mskt, )*imgtt.shape[-1], axis=-1)
                imgt[mskt3] = imgtt.ravel()
            else:
                imgtt = imgt_pad.ravel()
                imgtt = imgtt[:(map_xt.size)]
                imgt[mskt] = imgtt.ravel()
        to_render = to_render & (~mskt)
        if seed_indices.size > 0:
            sel_idx = to_render[seed_indices[:,0], seed_indices[:,1]]
            seed_indices = seed_indices[sel_idx]
    return imgt


def masked_dog_filter(img, sigma, mask=None, signed=True):
    """
    apply Difference of Gaussian filter to an image. if a mask is provided, make
    sure any signal outside the mask will not bleed out.
    Args:
        img (ndarray): C x H x W.
        sigma (float): standard deviation of first Gaussian kernel.
        mask: region that should be kept. H x W
    """
    sigma0, sigma1 = sigma, 2 * sigma
    if not np.issubdtype(img.dtype, np.floating):
        img = img.astype(np.float32)
    img0f = gaussian_filter1d(gaussian_filter1d(img, sigma0, axis=-1, mode='nearest'), sigma0, axis=-2, mode='nearest')
    img1f = gaussian_filter1d(gaussian_filter1d(img, sigma1, axis=-1, mode='nearest'), sigma1, axis=-2, mode='nearest')
    imgf = img0f - img1f
    if (mask is not None) and (not np.all(mask, axis=None)):
        mask_img = img.ptp() * (mask == 0)
        mask0f = gaussian_filter1d(gaussian_filter1d(mask_img, sigma0, axis=-1, mode='nearest'), sigma0, axis=-2, mode='nearest')
        mask1f = gaussian_filter1d(gaussian_filter1d(mask_img, sigma1, axis=-1, mode='nearest'), sigma1, axis=-2, mode='nearest')
        maskf = np.maximum(mask0f, mask1f)
        imgf_a = np.abs(imgf)
        imgf_a = (imgf_a - maskf).clip(0, None)
        imgf = imgf_a * np.sign(imgf)
    if not signed:
        imgf = np.abs(imgf)
    return imgf


def divide_bbox(bbox, **kwargs):
    xmin, ymin, xmax, ymax = bbox
    ht = ymax - ymin
    wd = xmax - xmin
    block_size = kwargs.get('block_size', max(ht, wd))
    min_num_blocks = kwargs.get('min_num_blocks', 1)
    round_output = kwargs.get('round_output', True)
    shrink_factor = kwargs.get('shrink_factor', 1)
    if not hasattr(block_size, '__len__'):
        block_size = (block_size, block_size)
    if not hasattr(min_num_blocks, '__len__'):
        min_num_blocks = (min_num_blocks, min_num_blocks)
    Nx = max(np.ceil(wd / block_size[1]), min_num_blocks[1])
    Ny = max(np.ceil(ht / block_size[0]), min_num_blocks[0])
    dx = int(np.ceil(wd / Nx))
    dy = int(np.ceil(ht / Ny))
    xt = np.linspace(xmin, xmax-dx, num=int(Nx), endpoint=True)
    yt = np.linspace(ymin, ymax-dy, num=int(Ny), endpoint=True)
    if shrink_factor != 1:
        dx_new = dx * shrink_factor
        dy_new = dy * shrink_factor
        xt = xt + (dx - dx_new)/2
        yt = yt + (dy - dy_new)/2
        dx = int(np.ceil(dx_new))
        dy = int(np.ceil(dy_new))
    if round_output:
        xt = np.round(xt).astype(np.int32)
        yt = np.round(yt).astype(np.int32)
    xx, yy = np.meshgrid(xt, yt)
    return xx.ravel(), yy.ravel(), xx.ravel() + dx, yy.ravel() + dy


def intersect_bbox(bbox0, bbox1):
    xmin = max(bbox0[0], bbox1[0])
    ymin = max(bbox0[1], bbox1[1])
    xmax = min(bbox0[2], bbox1[2])
    ymax = min(bbox0[3], bbox1[3])
    return (xmin, ymin, xmax, ymax), (xmin < xmax) and (ymin < ymax)


def find_elements_in_array(array, elements, tol=0):
    # if find elements in array, return indices, otherwise return -1
    shp = elements.shape
    array = array.ravel()
    elements = elements.ravel()
    sorter = array.argsort()
    idx = np.searchsorted(array, elements, sorter=sorter)
    idx = sorter[idx.clip(0, array.size-1)]
    neq = np.absolute(array[idx] - elements) > tol
    idx[neq] = -1
    return idx.reshape(shp)


def numpy_to_str_ascii(ar):
    t = ar.clip(0,255).astype(np.uint8).ravel()
    return t.tostring().decode('ascii')


def str_to_numpy_ascii(s):
    t =  np.frombuffer(s.encode('ascii'), dtype=np.uint8)
    return t


def load_plugin(plugin_name):
    modl, plugname = plugin_name.rsplit('.', 1)
    plugin_mdl = importlib.import_module(modl)
    plugin = getattr(plugin_mdl, plugname)
    return plugin


def hash_numpy_array(ar):
    if isinstance(ar, np.ndarray):
        return hash(ar.data.tobytes())
    elif isinstance(ar, list):
        return hash(tuple(ar))
    else:
        return hash(ar)


def indices_to_bool_mask(indx, size=None):
    if isinstance(indx, np.ndarray) and indx.dtype==bool:
        return indx
    if size is None:
        size = np.max(indx)
    mask = np.zeros(size, dtype=bool)
    mask[indx] = True
    return mask


def crop_image_from_bbox(img, bbox_img, bbox_out, **kwargs):
    """
    Crop an image based on the bounding box
    Args:
        img (np.ndarray): input image to be cropped.
        bbox_img: bounding box of the input image. [xmin, ymin, xmax, ymax]
        bbox_out: bounding box of the output image. [xmin, ymin, xmax, ymax]
    Kwargs:
        return_index (bool): if True, return the overlapping region of bbox_img
            and bbox_out & the slicings to position the overlapping region onto
            the output image; if False, return the output sized image without
            slicings.
        return_empty (bool): if False, return None if bbox_img and bbox_out not
            overlapping; if True, return an ndarray filled with fillval.
        fillval(scalar): fill values for invalid pixels in the output image.
    Return:
        imgout: output image. if return_indx is True, only return the overlap
            region between the two bboxes.
        index: the slicings to position the overlapping onto the output image.
            return only when return_index is True.
    """
    return_index = kwargs.get('return_index', False)
    return_empty = kwargs.get('return_empty', False)
    flip_indx = kwargs.get('flip_indx', False)
    fillval = kwargs.get('fillval', 0)
    x0 = bbox_img[0]
    y0 = bbox_img[1]
    if flip_indx:
        imght, imgwd = img.shape[1], img.shape[0]
    else:
        imght, imgwd = img.shape[0], img.shape[1]
    blkht = min(bbox_img[3] - bbox_img[1], imght)
    blkwd = min(bbox_img[2] - bbox_img[0], imgwd)
    outht = bbox_out[3] - bbox_out[1]
    outwd = bbox_out[2] - bbox_out[0]
    xmin = max(x0, bbox_out[0])
    xmax = min(x0 + blkwd, bbox_out[2])
    ymin = max(y0, bbox_out[1])
    ymax = min(y0 + blkht, bbox_out[3])
    if xmin >= xmax or ymin >= ymax:
        if return_index:
            return None, None
        else:
            if return_empty:
                outsz = [outht, outwd] + list(img.shape)[2:]
                imgout = np.full_like(img, fillval, shape=outsz)
                return imgout
            else:
                return None
    if flip_indx:
        dims = list(range(len(img.shape)))
        dims[:2] = [1,0]
        cropped = img[(xmin-x0):(xmax-x0), (ymin-y0):(ymax-y0), ...].transpose(dims)
    else:
        cropped = img[(ymin-y0):(ymax-y0), (xmin-x0):(xmax-x0), ...]
    dimpad = len(img.shape) - 2
    indx = tuple([slice(ymin-bbox_out[1], ymax-bbox_out[1]), slice(xmin-bbox_out[0],xmax-bbox_out[0])] +
            [slice(0, None)] * dimpad)
    if return_index:
        return cropped, indx
    else:
        outsz = [outht, outwd] + list(img.shape)[2:]
        imgout = np.full_like(img, fillval, shape=outsz)
        imgout[indx] = cropped
        return imgout


def chain_segment_rings(segments, directed=True, conn_lable=None) -> list:
    """
    Given id pairs of line segment points, assemble them into (closed) chains.
    Args:
        segments (Nsegx2 ndarray): vertices' ids of each segment. Each segment
            should only appear once, and the rings should be simple (no self
            intersection).
        directed (bool): whether the segments provided are directed. Default to
            True.
        conn_label (np.ndarray): preset groupings of the segments. If set to
            None, use the connected components from vertex adjacency.
    """
    inv_map, seg_n = np.unique(segments, return_inverse=True)
    seg_n = seg_n.reshape(segments.shape)
    if not directed:
        seg_n = np.sort(seg_n, axis=-1)
    Nseg = seg_n.shape[0]
    Npts = inv_map.size
    chains = []
    if conn_lable is None:
        A = sparse.csr_matrix((np.ones(Nseg), (seg_n[:,0], seg_n[:,1])), shape=(Npts, Npts))
        N_conn, V_conn = csgraph.connected_components(A, directed=directed, return_labels=True)
    else:
        u_lbl, S_conn = np.unique(conn_lable,  return_inverse=True)
        N_conn = u_lbl.size
        A = sparse.csc_matrix((S_conn+1, (seg_n[:,0], seg_n[:,1])), shape=(Npts, Npts))
    for n in range(N_conn):
        if conn_lable is None:
            vtx_mask = V_conn == n
            An = A[vtx_mask][:, vtx_mask]
        else:
            An0 = A == (n+1)
            An0.eliminate_zeros()
            vtx_mask = np.zeros(Npts, dtype=bool)
            sidx = np.unique(seg_n[S_conn == n], axis=None)
            vtx_mask[sidx] = True
            An = An0[vtx_mask][:, vtx_mask]
        vtx_idx = np.nonzero(vtx_mask)[0]
        while An.max() > 0:
            idx0, idx1 = np.unravel_index(np.argmax(An), An.shape)
            An[idx0, idx1] = 0
            An.eliminate_zeros()
            dis, pred = csgraph.shortest_path(An, directed=directed, return_predecessors=True, indices=idx1)
            if dis[idx0] < 0:
                raise ValueError('segment rings not closed.')
            seq = [idx0]
            crnt_node = idx0
            while True:
                crnt_node = pred[crnt_node]
                if crnt_node >= 0:
                    seq.insert(0, crnt_node)
                else:
                    break
            chain_idx = vtx_idx[seq]
            chains.append(inv_map[chain_idx])
            covered_edges = np.stack((seq[:-1], seq[1:]), axis=-1)
            if not directed:
                covered_edges = np.sort(covered_edges, axis=-1)
            R = sparse.csr_matrix((np.ones(len(seq)-1), (covered_edges[:,0], covered_edges[:,1])), shape=An.shape)
            An = An - R
    return chains


def signed_area(vertices, triangles) -> np.ndarray:
    tripts = vertices[triangles]
    v0 = tripts[:,1,:] - tripts[:,0,:]
    v1 = tripts[:,2,:] - tripts[:,1,:]
    return np.cross(v0, v1)


def expand_image(img, target_size, slices, fillval=0):
    if len(img.shape) == 3:
        target_size = list(target_size) + [img.shape[-1]]
    img_out = np.full_like(img, fillval, shape=target_size)
    img_out[slices[0], slices[1], ...] = img
    return img_out


def bbox_centers(bboxes):
    bboxes = np.array(bboxes, copy=False)
    cntr = 0.5 * bboxes @ np.array([[1,0],[0,1],[1,0],[0,1]]) - 0.5
    return cntr


def bbox_sizes(bboxes):
    bboxes = np.array(bboxes, copy=False)
    szs = bboxes @ np.array([[0,-1],[-1,0],[0,1],[1,0]])
    return szs.clip(0, None)


def bbox_intersections(bboxes0, bboxes1):
    xy_min = np.maximum(bboxes0[...,:2], bboxes1[...,:2])
    xy_max = np.minimum(bboxes0[...,-2:], bboxes1[...,-2:])
    bbox_int = np.concatenate((xy_min, xy_max), axis=-1)
    width = np.min(xy_max - xy_min, axis=-1)
    return bbox_int, width


def bbox_union(bboxes):
    bboxes = np.array(bboxes, copy=False)
    bboxes = bboxes.reshape(-1, 4)
    xy_min = bboxes[:,:2].min(axis=0)
    xy_max = bboxes[:,-2:].max(axis=0)
    return np.concatenate((xy_min, xy_max), axis=None)


def bbox_enlarge(bboxes, margin=0):
    return np.array(bboxes, copy=False) + np.array([-margin, -margin, margin, margin])


def parse_coordinate_files(filename, **kwargs):
    """
    parse a coordinate txt file. Each row in the file follows the pattern:
        image_path  x_min  y_min  x_max(optional)  y_max(optional)
    if x_max and y_max is not provided, they are inferred from tile_size.
    If relative path is provided in the image_path colume, at the first line
    of the file, the root_dir can be defined as:
        {ROOT_DIR}  rootdir_to_the_path
        {TILE_SIZE} tile_height tile_width
    Args:
        filename(str): full path to the coordinate file.
    Kwargs:
        rootdir: if the imgpaths colume in the file is relative paths, can
            use this to prepend the paths. Set to None to disable.
        tile_size: the tile size used to compute the bounding boxes in the
            absense of x_max and y_max in the file. If None, will read an
            image file to figure out
        delimiter: the delimiter to separate each colume in the file. If set
            to None, any whitespace will be considered.
    """
    root_dir = kwargs.get('root_dir', None)
    tile_size = kwargs.get('tile_size', None)
    delimiter = kwargs.get('delimiter', '\t') # None for any whitespace
    resolution = kwargs.get('resolution', None)
    imgpaths = []
    bboxes = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    if len(lines) == 0:
        raise RuntimeError(f'empty file: {filename}')
    start_line = 0
    for line in lines:
        if '{ROOT_DIR}' in line:
            start_line += 1
            tlist = line.strip().split(delimiter)
            if len(tlist) >= 2:
                root_dir = tlist[1]
        elif '{TILE_SIZE}' in line:
            start_line += 1
            tlist = line.strip().split(delimiter)
            if len(tlist) == 2:
                tile_size = (int(tlist[1]), int(tlist[1]))
            elif len(tlist) > 2:
                tile_size = (int(tlist[1]), int(tlist[2]))
            else:
                continue
        elif '{RESOLUTION}' in line:
            start_line += 1
            tlist = line.strip().split(delimiter)
            if len(tlist) >= 2:
                resolution = float(tlist[1])
        else:
            break
    relpath = bool(root_dir)
    for line in lines[start_line:]:
        line = line.strip()
        tlist = line.split(delimiter)
        if len(tlist) < 3:
            raise RuntimeError(f'corrupted coordinate file: {filename}')
        mpath = tlist[0]
        x_min = float(tlist[1])
        y_min = float(tlist[2])
        if (len(tlist) >= 5) and (tile_size is None):
            x_max = float(tlist[3])
            y_max = float(tlist[4])
        else:
            if tile_size is None:
                if relpath:
                    mpath_f = os.path.join(root_dir, mpath)
                else:
                    mpath_f = mpath
                img = imread(mpath_f, flag=cv2.IMREAD_GRAYSCALE)
                tile_size = img.shape
            x_max = x_min + tile_size[-1]
            y_max = y_min + tile_size[0]
        imgpaths.append(mpath)
        bboxes.append((x_min, y_min, x_max, y_max))
    return imgpaths, bboxes, root_dir, resolution


def rearrange_section_order(section_list, section_order_file, order_file_only=True, merge=False):
    if os.path.isfile(section_order_file):
        with open(section_order_file, 'r') as f:
            section_orders0 = f.readlines()
        section_orders = []
        z_lut = {}
        for k, s in enumerate(section_orders0):
            s = s.strip()
            if '\t' in s:
                secname = s.split('\t')[1]
                section_orders.append(secname)
                z_lut[secname] = int(s.split('\t')[0])
            else:
                section_orders.append(s)
                z_lut[s] = k
        assert len(section_orders) == len(set(section_orders))
        secnames = [os.path.splitext(os.path.basename(fname))[0] for fname in section_list]
        section_lut = {secname:fname for secname, fname in zip(secnames, section_list)}
        section_orders = [s for s in section_orders if s in secnames]
        if order_file_only:
            section_list_out = [section_lut[s] for s in section_orders if s in section_lut]
            z_indices = [z_lut[s] for s in section_orders if s in section_lut]
            return section_list_out, np.array(z_indices)
        if merge:
            secnames = sorted(list(set(secnames + section_orders)))
        section_orders_set = set(section_orders)
        sec_keys = []
        order_cnt = 0
        z_indices = np.arange(len(secnames))
        for k, s in enumerate(secnames):
            if s in section_orders_set:
                secname = section_orders[order_cnt]
                sec_keys.append(secname)
                z_indices[k] = z_lut[secname]
                order_cnt += 1
            else:
                sec_keys.append(s)
        return [section_lut.get(s, None) for s in sec_keys], z_indices
    else:
        return section_list, np.arange(len(section_list))
