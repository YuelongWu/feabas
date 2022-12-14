import collections
import importlib
from scipy import sparse
import scipy.sparse.csgraph as csgraph
import gc

import numpy as np


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
    fillval = kwargs.get('fillval', 0)
    x0 = bbox_img[0]
    y0 = bbox_img[1]
    blkht = min(bbox_img[3] - bbox_img[1], img.shape[0])
    blkwd = min(bbox_img[2] - bbox_img[0], img.shape[1])
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


##--------------------------------- caches -----------------------------------##

class Node:
    """
    Node used in doubly linked list.
    """
    def __init__(self, key, data):
        self.key = key # harshable key for indexing
        self.data = data
        self.pointer = None  # store e.g. pointer to freq node
        self.prev = None
        self.next = None


    def modify_data(self, data):
        self.data = data



class DoublyLinkedList:
    """
    Doubly linked list for LFU cache etc.
    Args:
        item(tuple): (key, data) pair of the first node. Return empty list if
            set to None.
    """
    def __init__(self, item=None):
        if item is None:
            self.head = None
            self.tail = None
            self._number_of_nodes = 0
        else:
            if isinstance(item, Node):
                first_node = item
            else:
                first_node = Node(*item)
            self.head = first_node
            self.tail = first_node
            self._number_of_nodes = 1


    def __len__(self):
        return self._number_of_nodes


    def clear(self):
        # Traverse the list to break reference cycles
        while self.head is not None:
            self.remove_head()


    def insert_before(self, node, item):
        if isinstance(item, Node):
            new_node = item
        else:
            new_node = Node(*item)
        if node is None:
            # empty list
            self.head = new_node
            self.tail = new_node
        else:
            prevnode = node.prev
            new_node.prev = prevnode
            new_node.next = node
            node.prev = new_node
            if prevnode is None:
                self.head = new_node
            else:
                prevnode.next = new_node
        self._number_of_nodes += 1


    def insert_after(self, node, item):
        if isinstance(item, Node):
            new_node = item
        else:
            new_node = Node(*item)
        if node is None:
            # empty list
            self.head = new_node
            self.tail = new_node
        else:
            nextnode = node.next
            new_node.next = nextnode
            new_node.prev = node
            node.next = new_node
            if nextnode is None:
                self.tail = new_node
            else:
                nextnode.prev = new_node
        self._number_of_nodes += 1


    def pop_node(self, node):
        if node is None:
            return None
        prevnode = node.prev
        nextnode = node.next
        if prevnode is not None:
            prevnode.next = nextnode
        else:
            self.head = nextnode
        if nextnode is not None:
            nextnode.prev = prevnode
        else:
            self.tail = prevnode
        node.prev = None
        node.next = None
        self._number_of_nodes -= 1
        return node


    def remove_node(self, node):
        del node.key
        del node.data
        del node.pointer
        self.pop_node(node)


    def insert_head(self, item):
        self.insert_before(self.head, item)


    def insert_tail(self, item):
        self.insert_after(self.tail, item)


    def pop_head(self):
        return self.pop_node(self.head)


    def pop_tail(self):
        return self.pop_node(self.tail)


    def remove_head(self):
        self.remove_node(self.head)


    def remove_tail(self):
        self.remove_node(self.tail)



class CacheNull:
    """
    Cache class with no capacity. Mostlys to define Cache APIs.
    Attributes:
        _maxlen: the maximum capacity of the cache. No upper limit if set to None.
    """
    def __init__(self, maxlen=0):
        self._maxlen = maxlen

    def clear(self, instant_gc=False):
        """Clear cache"""
        if instant_gc:
            gc.collect()

    def item_accessed(self, key_list):
        """Add accessed time by 1 for items in key list (used for freq record)"""
        pass

    def __contains__(self, key):
        """Check item availability in the cache"""
        return False

    def __getitem__(self, key):
        """Access an item"""
        errmsg = "fail to access data from empty cache"
        raise NotImplementedError(errmsg)

    def __len__(self):
        """Current number of items in the cache"""
        return 0

    def __setitem__(self, key, data):
        """Cache an item"""
        pass

    def update_item(self, key, data):
        """force update a cached item"""
        pass

    def _evict_item_by_key(self, key):
        """remove an item by providing the key"""
        pass


class CacheFIFO(CacheNull):
    """
    Cache with first in first out (FIFO) replacement policy.
    """
    def __init__(self, maxlen=None):
        self._maxlen = maxlen
        self._keys = collections.deque(maxlen=maxlen)
        self._vals = collections.deque(maxlen=maxlen)


    def clear(self, instant_gc=False):
        self._keys.clear()
        self._vals.clear()
        if instant_gc:
            gc.collect()


    def __contains__(self, key):
        return key in self._keys


    def __getitem__(self, key):
        if key in self._keys:
            indx = self._keys.index(key)
            return self._vals[indx]
        else:
            errmsg = "fail to access data with key {} from cached.".format(key)
            raise KeyError(errmsg)


    def __len__(self):
        return len(self._keys)


    def __setitem__(self, key, data):
        if (self._maxlen) == 0 or (key in self._keys):
            return
        self._keys.append(key)
        self._vals.append(data)


    def update_item(self, key, data):
        if (self._maxlen) == 0:
            return
        if key in self._keys:
            indx = self._keys.index(key)
            self._vals[indx] = data
        else:
            self.__setitem__(key, data)



class CacheLRU(CacheNull):
    """
    Cache with least recently used (LRU) replacement policy
    """
    def __init__(self, maxlen=None):
        self._maxlen = maxlen
        self._cached_nodes = {}
        self._cache_list = DoublyLinkedList() # head:old <-> tail:new


    def clear(self, instant_gc=False):
        self._cached_nodes.clear()
        self._cache_list.clear()
        if instant_gc:
            gc.collect()


    def item_accessed(self, key_list):
        for key in key_list:
            self._move_item_to_tail(key)


    def _evict_item_by_key(self, key):
        if key in self._cached_nodes:
            node = self._cached_nodes.pop(key)
            self._cache_list.remove_node(node)


    def _evict_item_by_policy(self):
        node = self._cache_list.head
        if node is not None:
            key = node.key
            self._evict_item_by_key(key)


    def _move_item_to_tail(self, key):
        if key in self._cached_nodes:
            node = self._cached_nodes[key]
            if node.next is None:
                return
            node = self._cache_list.pop_node(node)
            self._cache_list.insert_tail(node)


    def __contains__(self, key):
        return key in self._cached_nodes


    def __getitem__(self, key):
        if key in self._cached_nodes:
            node = self._cached_nodes[key]
            self._move_item_to_tail(key)
            return node.data
        else:
            errmsg = "fail to access data with key {} from cached.".format(key)
            raise KeyError(errmsg)


    def __len__(self):
        return len(self._cached_nodes)


    def __setitem__(self, key, data):
        if (self._maxlen == 0) or (key in self._cached_nodes):
            return
        if self._maxlen is not None:
            while len(self._cached_nodes) >= self._maxlen:
                self._evict_item_by_policy()
        data_node = Node(key, data)
        self._cache_list.insert_tail(data_node)
        self._cached_nodes[key] = data_node


    def update_item(self, key, data):
        if self._maxlen == 0:
            return
        if key in self._cached_nodes:
            data_node = self._cached_nodes[key]
            data_node.modify_data(data)
        else:
            self.__setitem__(key, data)



class CacheLFU(CacheNull):
    """
    Cache with least frequent used (LFU) replacement policy.
    Attributes:
        _cached_nodes(dict): dictionary holding the data nodes.
        _freq_list(DoublyLinkedList): frequecy list, with each node holding
            accessed frequency and pointing to a DoublyLinkedList holding
            cached data nodes, with later added nodes attached to the tail. Each
            data node contains cached data and points to its frequency node.
    """
    def __init__(self, maxlen=None):
        self._maxlen = maxlen
        self._cached_nodes = {}
        self._freq_list = DoublyLinkedList()


    def clear(self, instant_gc=False):
        for key in self._cached_nodes:
            self._evict_item_by_key(key)
        self._freq_list.clear()
        if instant_gc:
            gc.collect()


    def item_accessed(self, key_list):
        for key in key_list:
            self._increase_item_access_number_by_one(key)


    def _evict_item_by_key(self, key):
        if key in self._cached_nodes:
            node = self._cached_nodes.pop(key)
            freq_node = node.pointer
            cache_list = freq_node.pointer
            cache_list.remove_node(node)
            if (len(cache_list) == 0) and (freq_node.data != 0):
                self._freq_list.remove_node(freq_node)


    def _evict_item_by_policy(self):
        freq_node = self._freq_list.head
        while freq_node is not None:
            cache_list = freq_node.pointer
            if len(cache_list) > 0:
                key = cache_list.head.key
                self._evict_item_by_key(key)
                break
            else:
                freq_node = freq_node.next


    def _increase_item_access_number_by_one(self, key):
        if key in self._cached_nodes:
            node = self._cached_nodes[key]
            freq_node = node.pointer
            cache_list = freq_node.pointer
            cnt = freq_node.data
            if (freq_node.next is None) or (freq_node.next.data != cnt + 1):
                if len(cache_list) == 1:
                    # only this data node linked to the freq node.
                    freq_node.data += 1
                    return
                else:
                    self._freq_list.insert_after(freq_node, (None, cnt+1))
                    freq_node.next.pointer = DoublyLinkedList()
            target_cache_list = freq_node.next.pointer
            node = cache_list.pop_node(node)
            node.pointer = freq_node.next
            if (len(cache_list) == 0) and (freq_node.data != 0):
                self._freq_list.remove_node(freq_node)
            target_cache_list.insert_tail(node)


    def __contains__(self, key):
        return key in self._cached_nodes


    def __getitem__(self, key):
        if key in self._cached_nodes:
            node = self._cached_nodes[key]
            self._increase_item_access_number_by_one(key)
            return node.data
        else:
            errmsg = "fail to access data with key {} from cached.".format(key)
            raise KeyError(errmsg)

    def __len__(self):
        return len(self._cached_nodes)


    def __setitem__(self, key, data):
        if (self._maxlen == 0) or (key in self._cached_nodes):
            return
        if self._maxlen is not None:
            while len(self._cached_nodes) >= self._maxlen:
                self._evict_item_by_policy()
        if (self._freq_list.head is None) or (self._freq_list.head.data != 0):
            self._freq_list.insert_head((None, 0))
            self._freq_list.head.pointer = DoublyLinkedList()
        data_node = Node(key, data)
        freq_node = self._freq_list.head
        data_node.pointer = freq_node
        cache_list = freq_node.pointer
        cache_list.insert_tail(data_node)
        self._cached_nodes[key] = data_node


    def update_item(self, key, data):
        if self._maxlen == 0:
            return
        if key in self._cached_nodes:
            data_node = self._cached_nodes[key]
            data_node.modify_data(data)
        else:
            self.__setitem__(key, data)


class CacheMFU(CacheLFU):
    """
    Cache with most frequent used replacement policy.
    This policy could be useful in applications like rendering when the purpose
    is to cover the entire dataset once, and data already accessed multiple times
    is less likely to be accessed again.
    """
    def _evict_item_by_policy(self):
        freq_node = self._freq_list.tail
        while freq_node is not None:
            cache_list = freq_node.pointer
            if len(cache_list) > 0:
                key = cache_list.head.key
                self._evict_item_by_key(key)
                break
            else:
                freq_node = freq_node.prev


def generate_cache(cache_type='fifo', maxlen=None):
    if (maxlen == 0) or (cache_type.lower() == 'none'):
        return CacheNull()
    elif cache_type.lower() == 'fifo':
        return CacheFIFO(maxlen=maxlen)
    elif cache_type.lower() == 'lru':
        return CacheLRU(maxlen=maxlen)
    elif cache_type.lower() == 'lfu':
        return CacheLFU(maxlen=maxlen)
    elif cache_type.lower() == 'mfu':
        return CacheMFU(maxlen=maxlen)
    else:
        errmsg = 'cache type {} not implemented'.format(cache_type)
        raise NotImplementedError(errmsg)
