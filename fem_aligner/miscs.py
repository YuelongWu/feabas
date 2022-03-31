import collections


class DequeDict:
    """
    Dictionary with fixed max length. Used for caching.
    """
    def __init__(self, maxlen=None):
        self._keys = collections.deque(maxlen=maxlen)
        self._vals = collections.deque(maxlen=maxlen)


    def clear(self):
        self._keys.clear()
        self._vals.clear()


    def __contains__(self, key):
        return key in self._keys


    def __getitem__(self, key):
        if key in self._keys:
            indx = self._keys.index(key)
            return self._vals[indx]
        else:
            raise KeyError(key)


    def __len__(self):
        return len(self._keys)


    def __setitem__(self, key, value):
        if not key in self._keys:
            self._keys.append(key)
            self._vals.append(value)
