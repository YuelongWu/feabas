"""
Some utility functions to interact with storages Google Cloud.
"""
import glob
from functools import lru_cache
import os
import yaml


LOCAL_TEMP_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp')

@lru_cache(maxsize=1)
def GCP_client():
    from google.cloud import storage
    return storage.Client()


def GCP_parse_object_name(url):
    plist = url.replace('gs://', '').split('/')
    plist = [s for s in plist if s]
    bucket = plist[0]
    relpath = '/'.join(plist[1:])
    return bucket, relpath


def GCP_get_blob(url):
    bucketname, relpath = GCP_parse_object_name(url)
    client = GCP_client()
    bucket = client.get_bucket(bucketname)
    blob = bucket.blob(relpath)
    return blob


def parse_file_driver(filename):
    if filename.startswith('gs://'):
        driver = 'gs'
    elif filename.startswith('s3://'):
        driver = 's3'
    else:
        if filename.startswith('file://'):
            filename = filename[7:]
        driver = 'file'
    return driver, filename


def load_yaml(filename):
    if not file_exists(filename):
        d = {}
    else:
        with File(filename, 'r') as f:
            d = yaml.safe_load(f)
    return d


def list_folder_content(pathname, recursive=False):
    if pathname is None:
        return []
    driver, pathname = parse_file_driver(pathname)
    if driver == 'file':
        flist = glob.glob(pathname, recursive=recursive)
    elif driver == 'gs':
        bucketname, relpath = GCP_parse_object_name(pathname)
        prefix = relpath.split('*')[0]
        ext = relpath.split('*')[-1]
        delimiter = None if recursive else '/'
        blobs = GCP_client().list_blobs(bucketname, prefix=prefix, delimiter=delimiter)
        flist = []
        for b in blobs:
            fname = b.name
            if not fname.endswith(ext):
                continue
            flist.append(join_paths('gs://'+bucketname, fname))
    return flist


def file_exists(filename, use_cache=False):
    if filename is None:
        return False
    if use_cache:
        dir_name, b_name = os.path.split(filename)
        bname_no_ext = os.path.splitext(b_name)[0]
        b_name = b_name.replace(bname_no_ext, '*')
        prefix = join_paths(dir_name, b_name)
        file_list = _cached_folder_contents(prefix)
        return (filename in file_list)
    else:
        driver, filename = parse_file_driver(filename)
        if driver == 'gs':
            blob = GCP_get_blob(filename)
            return blob.exists()
        else:
            return os.path.isfile(filename)

@lru_cache(maxsize=5)
def _cached_folder_contents(pathname):
    return set(list_folder_content(pathname))

def dir_exists(dirname):
    driver, dirname = parse_file_driver(dirname)
    if driver == 'gs':
        bucketname, relpath = GCP_parse_object_name(dirname)
        blobs = GCP_client().list_blobs(bucketname, prefix=relpath)
        return blobs.num_results > 0
    else:
        return os.path.isdir(dirname)


def join_paths(*args):
    if any([s is None for s in args]):
        return None
    parent_dir = args[0]
    if parent_dir.startswith('gs://'):
        pth = '/'.join(args)
    else:
        pth = os.path.join(*args)
    return pth


def expand_dir(pth):
    driver, pth = parse_file_driver(pth)
    if driver == 'file':
        pth = os.path.abspath(os.path.expanduser(os.path.expandvars(pth)))
    return pth


def makedirs(filename, exist_ok=True):
    driver, filename = parse_file_driver(filename)
    if driver == 'file':
        os.makedirs(filename, exist_ok=exist_ok)


def remove_file(filename):
    if not file_exists(filename):
        return False
    driver, filename = parse_file_driver(filename)
    if driver == 'gs':
        blob = GCP_get_blob(filename)
        blob.delete()
    else:
        os.remove(filename)
    return True


def h5file_class():
    import h5py # delay import to delay numpy, for thread control
    class H5File(h5py.File):
        """
        if the target file is on cloud, interact locally and then transfer to cloud.
        """
        def __init__(self, name, *args, **kwargs):
            driver, name = parse_file_driver(name)
            if driver == 'gs':
                blob = GCP_get_blob(name)
                self._blob = blob
                tmpname = hex(os.getpid())[2:] + '_' + hex(id(self))[2:] + '.h5'
                self._localname = os.path.join(LOCAL_TEMP_FOLDER, tmpname)
                if self._blob.exists():
                    os.makedirs(LOCAL_TEMP_FOLDER, exist_ok=True)
                    self._blob.download_to_filename(self._localname)
                self._type = 'gs'
            else:
                self._blob = None
                self._localname = name
                self._type = 'file'
            super().__init__(self._localname, *args, **kwargs)

        def close(self):
            mode = self.mode
            super().close()
            if (mode == 'r+') and (self._type == 'gs'):
                self._blob.upload_from_filename(self._localname)
            if (self._blob is not None) and os.path.isfile(self._localname):
                os.remove(self._localname)
    return H5File


class File():
    def __init__(self, filename, mode='r'):
        driver, filename = parse_file_driver(filename)
        self._driver = driver
        self._filename = filename
        self._mode = mode
        self._file = None

    def open(self):
        if self._driver == 'file':
            self._file = open(self._filename, self._mode)
        elif self._driver == 'gs':
            blob = GCP_get_blob(self._filename)
            self._file = blob.open(mode=self._mode)
        return self._file

    def close(self):
        if self._file:
            self._file.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()