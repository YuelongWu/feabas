"""
Some utility functions to interact with storages Google Cloud.
"""
import getpass
import glob
from functools import lru_cache
import io
import os
import tempfile
import yaml


def _local_temp_folder():
    """
    local folder to stage GCS transfers in: FEABAS_TMPDIR if set, otherwise a
    per-user folder (the system temp folder may be shared, e.g. /tmp on
    Linux) in the system temp folder, which follows TMPDIR/TEMP/TMP. Falls
    back to <package parent>/temp if that is not writable.
    """
    candidates = []
    if os.environ.get('FEABAS_TMPDIR'):
        candidates.append(os.environ['FEABAS_TMPDIR'])
    try:
        user = getpass.getuser()
    except Exception:
        user = 'user'
    try:
        candidates.append(os.path.join(tempfile.gettempdir(), 'feabas_' + user))
    except Exception:
        pass
    candidates.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'temp'))
    for dirname in candidates:
        try:
            os.makedirs(dirname, exist_ok=True)
            if os.access(dirname, os.W_OK):
                return dirname
        except OSError:
            continue
    return candidates[-1]

LOCAL_TEMP_FOLDER = _local_temp_folder()


def local_temp_file(suffix='', dirname=None, prefix='feabas_'):
    """
    name of a new empty local file for staging (closed, so that other
    libraries can open it on any platform). In LOCAL_TEMP_FOLDER by default.
    Created with the default permissions (umask), unlike tempfile.mkstemp,
    as staged outputs are renamed into place.
    """
    import uuid
    if dirname is None:
        dirname = LOCAL_TEMP_FOLDER
    os.makedirs(dirname, exist_ok=True)
    while True:
        fname = os.path.join(dirname, prefix + uuid.uuid4().hex + suffix)
        try:
            with open(fname, 'x'):
                pass
            return fname
        except FileExistsError:
            continue


def _remove_quietly(fname):
    try:
        os.remove(fname)
    except OSError:
        pass

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
    bucket = client.bucket(bucketname)  # no request, unlike get_bucket
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

def file_md5(filename):
    """base64 MD5 of a file's content (from the object metadata on GCS), None if missing."""
    import base64, hashlib
    if filename is None:
        return None
    driver, filename = parse_file_driver(filename)
    if driver == 'gs':
        from google.api_core.exceptions import NotFound
        blob = GCP_get_blob(filename)
        try:
            blob.reload()
        except NotFound:
            return None
        if blob.md5_hash is not None:
            return blob.md5_hash
        return 'crc32c:' + blob.crc32c   # composite objects have no MD5
    if not os.path.isfile(filename):
        return None
    h = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def read_json(filename, default=None):
    """content of a json file, or default if it does not exist (one request on GCS)."""
    import json
    if filename is None:
        return default
    driver, fname = parse_file_driver(filename)
    if driver == 'gs':
        from google.api_core.exceptions import NotFound
        try:
            return json.loads(GCP_get_blob(fname).download_as_bytes())
        except NotFound:
            return default
    try:
        with open(fname, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def write_json(filename, obj, gcs_metadata=None):
    """
    write obj to a json file. On GCS, gcs_metadata ({str: str}) is set as the
    object's custom metadata in the same (atomic) upload, so that it can be
    read back from a folder listing (list_gcs_metadata).
    """
    import json
    driver, fname = parse_file_driver(filename)
    if driver == 'gs':
        blob = GCP_get_blob(fname)
        if gcs_metadata is not None:
            blob.metadata = gcs_metadata
        blob.upload_from_string(json.dumps(obj), content_type='application/json')
    else:
        makedirs(os.path.dirname(fname))
        with open(fname, 'w') as f:
            json.dump(obj, f)


def list_gcs_metadata(dirname, ext=''):
    """
    {gs:// object name: custom metadata (None if not set)} for the objects
    directly in a GCS folder whose names end with ext, from one (paginated)
    listing.
    """
    bucketname, relpath = GCP_parse_object_name(dirname)
    prefix = relpath.rstrip('/') + '/' if relpath else ''
    out = {}
    for b in GCP_client().list_blobs(bucketname, prefix=prefix, delimiter='/'):
        if b.name.endswith(ext):
            out[join_paths('gs://'+bucketname, b.name)] = b.metadata
    return out


def map_io(func, items, num_threads=10):
    """
    list(map(func, items)) with I/O-bound func run in a thread pool (GCS
    requests or small file reads). 10 threads matches the default size of
    the GCS client's connection pool.
    """
    items = list(items)
    if (num_threads <= 1) or (len(items) <= 1):
        return [func(s) for s in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(num_threads, len(items))) as executor:
        return list(executor.map(func, items))


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
        pth = pth.replace('//','/').replace('gs:/','gs://')
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


_H5_WRITE_MODES = ('w', 'x', 'w-')     # create or truncate: nothing to read first


def h5file_class():
    import h5py # delay import to delay numpy, for thread control
    class H5File(h5py.File):
        """
        h5py.File that also works on GCS. The file is staged locally: the
        object is downloaded first unless the mode creates a new file, and
        a file opened for writing is uploaded when closed. Files created with
        mode 'w'/'x' are written to a temporary file and only appear (uploaded
        to GCS, or renamed locally) if the `with` block exits without an
        exception, so that a failed write never leaves a partial file that
        later stages would take as a finished output.
        """
        def __init__(self, name, mode='r', *args, **kwargs):
            driver, name = parse_file_driver(name)
            self._type = driver
            self._target = name
            self._blob = None
            self._staged = mode in _H5_WRITE_MODES
            self._finalized = False
            if driver == 'gs':
                self._blob = GCP_get_blob(name)
                self._localname = local_temp_file(suffix='.h5')
                if not self._staged:
                    try:
                        self._download(mode)
                    except Exception:
                        _remove_quietly(self._localname)
                        raise
                if mode in ('x', 'w-'):
                    mode = 'w'  # existence checked by the upload (if_generation_match)
                    self._exclusive = True
                else:
                    self._exclusive = False
            elif self._staged:
                if (mode in ('x', 'w-')) and os.path.exists(name):
                    raise FileExistsError(name)
                mode = 'w'
                self._localname = local_temp_file(suffix='.tmp', dirname=os.path.dirname(os.path.abspath(name)),
                                                   prefix='.'+os.path.basename(name)+'.')
            else:
                self._localname = name
            try:
                super().__init__(self._localname, mode, *args, **kwargs)
            except Exception:
                if self._localname != self._target:
                    _remove_quietly(self._localname)
                raise

        def _download(self, mode):
            from google.api_core.exceptions import NotFound
            try:
                self._blob.download_to_filename(self._localname)
            except NotFound:
                _remove_quietly(self._localname)
                if mode != 'a':     # mode 'a' creates the file
                    raise FileNotFoundError(self._target)

        def close(self, commit=True):
            if self._finalized:
                return
            self._finalized = True
            writable = bool(self.id) and (self.mode == 'r+')
            super().close()
            try:
                if commit and writable:
                    if self._type == 'gs':
                        from google.api_core.exceptions import PreconditionFailed
                        kw = {'if_generation_match': 0} if self._exclusive else {}
                        try:
                            self._blob.upload_from_filename(self._localname, **kw)
                        except PreconditionFailed:
                            raise FileExistsError(self._target)
                    elif self._staged:
                        os.replace(self._localname, self._target)
            finally:
                if self._localname != self._target:
                    _remove_quietly(self._localname)

        def __exit__(self, exc_type, exc_value, traceback):
            self.close(commit=exc_type is None)
    return H5File


class File():
    """
    open() for local and GCS files. Writes ('w' modes) go to a buffer (GCS)
    or a temporary file (local) and only replace the target if the `with`
    block exits without an exception.
    """
    def __init__(self, filename, mode='r'):
        driver, filename = parse_file_driver(filename)
        self._driver = driver
        self._filename = filename
        self._mode = mode
        self._file = None
        self._tmpname = None

    def open(self):
        writing = 'w' in self._mode
        if self._driver == 'file':
            if writing:
                self._tmpname = local_temp_file(suffix='.tmp', dirname=os.path.dirname(os.path.abspath(self._filename)),
                                                prefix='.'+os.path.basename(self._filename)+'.')
                self._file = open(self._tmpname, self._mode)
            else:
                self._file = open(self._filename, self._mode)
        elif self._driver == 'gs':
            if writing:
                self._file = io.BytesIO() if 'b' in self._mode else io.StringIO()
            else:
                self._file = GCP_get_blob(self._filename).open(mode=self._mode)
        return self._file

    def close(self, commit=True):
        if self._file is None:
            return
        f, self._file = self._file, None
        try:
            if (self._driver == 'gs') and ('w' in self._mode):
                if commit:
                    data = f.getvalue()
                    GCP_get_blob(self._filename).upload_from_string(data)
                f.close()
            else:
                f.close()
                if (self._tmpname is not None) and commit:
                    os.replace(self._tmpname, self._filename)
        finally:
            if self._tmpname is not None:
                _remove_quietly(self._tmpname)
                self._tmpname = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(commit=exc_type is None)