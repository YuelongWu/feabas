"""
Some utility functions to interact with Google Cloud.
"""
from functools import lru_cache
from google.cloud import storage
import h5py
import os


LOCAL_TEMP_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp')

@lru_cache(maxsize=1)
def GCP_client():
    return storage.Client()


class H5File(h5py.File):
    """
    if the target file is on cloud, interact locally and then transfer to cloud.
    """
    def __init__(self, name, *args, **kwargs):
        if name.startswith('gs://'):
            plist = name.replace('gs://', '').split('/')
            plist = [s for s in plist if s]
            bucket = plist[0]
            relpath = '/'.join(plist[1:])
            bucket = GCP_client().get_bucket(bucket)
            blob = bucket.blob(relpath)
            self._blob = blob
            tmpname = hex(os.getpid())[2:] + '_' + hex(id(self))[2:] + '.h5'
            self._localname = os.path.join(LOCAL_TEMP_FOLDER, tmpname)
        elif name.startswith('file://'):
            self._blob = None
            self._localname = name[7:]
        else:
            self._blob = None
            self._localname = name
        if isinstance(self._blob, storage.Blob):
            os.makedirs(LOCAL_TEMP_FOLDER, exist_ok=True)
            if self._blob.exists():
                self._blob.download_to_filename(self._localname)
        super().__init__(self._localname, *args, **kwargs)


    def close(self):
        mode = self.mode
        super().close()
        if (mode == 'r+') and isinstance(self._blob, storage.Blob):
            self._blob.upload_from_filename(self._localname)
        if self._blob is not None:
            os.remove(self._localname)