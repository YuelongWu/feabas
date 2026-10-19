"""
Record which stitching transformation each derived output was made from, so
that a stage can tell whether its existing outputs are still up to date
without relying on file timestamps (unreliable across machines and buckets).

Each section has a small json record per stage, written only by the job that
writes the section's outputs:
    stitch/provenance/<sec>.json: {mip: fingerprint of the stitch tform}
        for each stitched level (downsampled levels inherit their source's)
    thumbnail_align/provenance/<sec>.json: {'source': fingerprint} of the
        stitched level the thumbnail was generated from
Outputs without a record (made by older versions) are considered up to date.
On GCS, the record is also stored as the json object's custom metadata, so
that the records of all the sections are read from one folder listing
(load_all) instead of one request per section.
"""
from functools import lru_cache
import json
import os

from feabas import storage


GCS_METADATA_KEY = 'feabas_record'


@lru_cache(maxsize=None)
def tform_fingerprint(tform_name):
    """
    fingerprint of a stitching transformation file: the tform_id written with
    the meshes (stitcher.Stitcher.save_to_h5), or the content checksum. On
    GCS, the object checksum is used to avoid downloading the file. None if
    the file does not exist.
    """
    if not storage.file_exists(tform_name):
        return None
    driver, _ = storage.parse_file_driver(tform_name)
    if driver != 'gs':
        import h5py
        with h5py.File(tform_name, 'r') as f:
            tform_id = f.attrs.get('tform_id', None)
        if tform_id is not None:
            return 'id:' + (tform_id.decode() if isinstance(tform_id, bytes) else str(tform_id))
    return 'md5:' + storage.file_md5(tform_name)


def load(record_file):
    if record_file is None:
        return {}
    return storage.read_json(record_file, default={})


def load_all(record_files, num_threads=10):
    """
    {record file: record} for a list of record files. On GCS, the records
    are read from the listing of their folder(s) (object metadata), falling
    back to reading the json for objects without it; local files are read in
    a thread pool.
    """
    record_files = [s for s in record_files if s is not None]
    records = {}
    to_read = []
    gcs_dirs = sorted(set(os.path.dirname(s) for s in record_files
                          if storage.parse_file_driver(s)[0] == 'gs'))
    listed = {}
    for dirname in gcs_dirs:
        listed.update(storage.list_gcs_metadata(dirname, ext='.json'))
    listed = {storage.join_paths(s): m for s, m in listed.items()}
    for fname in record_files:
        meta = listed.get(storage.join_paths(fname), {})
        if storage.parse_file_driver(fname)[0] != 'gs':
            to_read.append(fname)
        elif storage.join_paths(fname) not in listed:
            records[fname] = {}
        elif (meta is not None) and (GCS_METADATA_KEY in meta):
            records[fname] = json.loads(meta[GCS_METADATA_KEY])
        else:
            to_read.append(fname)
    for fname, rec in zip(to_read, storage.map_io(load, to_read, num_threads=num_threads)):
        records[fname] = rec
    return records


def update(record_file, entries, record=None):
    """
    set (or remove, for None values) entries of a record file. record: its
    current content if already loaded (skip reading it again).
    """
    if record_file is None:
        return
    record = dict(load(record_file) if record is None else record)
    for key, val in entries.items():
        if val is None:
            record.pop(str(key), None)
        else:
            record[str(key)] = val
    storage.write_json(record_file, record, gcs_metadata={GCS_METADATA_KEY: json.dumps(record)})


def level_fingerprint(record_file, mip):
    """fingerprint recorded for a stitched mip level, None if unknown."""
    return load(record_file).get(str(mip), None)


def inherit_level(record_file, src_mip, tgt_mip):
    """record a level generated from another one: same source transformation."""
    if record_file is None:
        return
    record = load(record_file)
    update(record_file, {tgt_mip: record.get(str(src_mip), None)}, record=record)

