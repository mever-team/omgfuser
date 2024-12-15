"""
Created by Dimitrios Karageorgiou, email: dkarageo@iti.gr

Originally distributed under: https://github.com/mever-team/omgfuser

Copyright 2024 Media Analysis, Verification and Retrieval Group -
Information Technologies Institute - Centre for Research and Technology Hellas, Greece

This piece of code is licensed under the Apache License, Version 2.0.
A copy of the license can be found in the LICENSE file distributed together
with this file, as well as under https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under this repository is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the license for the specific language governing permissions and
limitations under the License.
"""

import csv
import hashlib
import io
import logging
import pathlib
from collections import Counter
from typing import Union, Optional

import click
import lmdb
import tqdm
import networkx as nx


__version__: str = "0.0.5-alpha"
__author__: str = "Dimitrios Karageorgiou"
__email__: str = "dkarageo@iti.gr"


class LMDBFileStorage:
    """A file storage for handling large datasets based on LMDB."""
    def __init__(self,
                 db_path: pathlib.Path,
                 map_size: int = 1024*1024*1024*1024,  # 1TB
                 read_only: bool = False,
                 max_readers: int = 128):
        self.db: lmdb.Environment = lmdb.open(
            str(db_path),
            map_size=map_size,
            subdir=False,
            readonly=read_only,
            max_readers=max_readers,
            lock=False,
            sync=False
        )

    def open_file(self, file_id: str, mode: str = "r") -> Union[io.TextIOWrapper, io.BytesIO]:
        """Returns a file-like stream of a file in the database."""
        # with self.db.begin() as trans:
        #     data: bytes = trans.get(file_id.encode("ascii"))
        with self.db.begin(buffers=True) as trans:
            data = trans.get(file_id.encode("ascii"))
        stream: io.BytesIO = io.BytesIO(data)

        if mode == "r":
            reader: io.TextIOWrapper = io.TextIOWrapper(stream)
        elif mode == "b":
            reader: io.BytesIO = stream
        else:
            raise RuntimeError(f"Unsupported file mode: '{mode}'. Only 'r' and 'b' are supported.")

        return reader

    def write_file(self, file_id: str, file_data: bytes) -> None:
        with self.db.begin(write=True) as trans:
            trans.put(file_id.encode("ascii"), file_data)

    def get_all_ids(self) -> list[str]:
        with self.db.begin() as trans:
            cursor = trans.cursor()
            ids: list[str] = [k for k, _ in cursor]
        return ids

    def close(self) -> None:
        self.db.close()


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("-c", "--csv_file", required=True,
              type=click.Path(dir_okay=False, exists=True, path_type=pathlib.Path),
              help="Path to a CSV file containing relative paths to the dataset files.")
@click.option("-b", "--base_dir",
              type=click.Path(file_okay=False, exists=True, path_type=pathlib.Path),
              help="Base directory of the dataset. Paths inside the CSV should be relative "
                   "to that path. When not provided, the directory of the CSV file is "
                   "considered as the base directory.")
@click.option("-o", "--output_file", required=True,
              type=click.Path(dir_okay=False, path_type=pathlib.Path),
              help="Path to the database. If the file does not "
                   "exist, a new database is generated. Otherwise, it should point to a "
                   "previous instance of the LMDB, where data will be added.")
def add_csv(
    csv_file: pathlib.Path,
    base_dir: Optional[pathlib.Path],
    output_file: pathlib.Path
) -> None:
    if base_dir is None:
        base_dir = csv_file.parent
    db: LMDBFileStorage = LMDBFileStorage(output_file)
    add_csv_to_db(csv_file, db, base_dir)
    db.close()


@cli.command()
@click.option("-s", "--src", required=True,
              type=click.Path(dir_okay=False, path_type=pathlib.Path, exists=True),
              help="Database whose files will be added to the destination database.")
@click.option("-d", "--dest", required=True,
              type=click.Path(dir_okay=False, path_type=pathlib.Path),
              help="Database where file from source database will be added.")
def add_db(
    src: pathlib.Path,
    dest: pathlib.Path
) -> None:
    """Adds all the contents of a database to another."""
    src_db: LMDBFileStorage = LMDBFileStorage(src, read_only=True)
    dest_db: LMDBFileStorage = LMDBFileStorage(dest)

    for k in tqdm.tqdm(src_db.get_all_ids(), desc="Copying files", unit="file"):
        k = str(k, 'UTF-8')
        src_file: io.BytesIO = src_db.open_file(k, mode="b")
        dest_db.write_file(k, src_file.read())

    src_db.close()
    dest_db.close()


@cli.command()
@click.option("-c", "--csv_file", required=True,
              type=click.Path(dir_okay=False, exists=True, path_type=pathlib.Path),
              help="Path to a CSV file containing relative paths to the dataset files.")
@click.option("-b", "--base_dir",
              type=click.Path(file_okay=False, exists=True, path_type=pathlib.Path),
              help="Base directory of the dataset. Paths inside the CSV should be relative "
                   "to that path. When not provided, the directory of the CSV file is "
                   "considered as the base directory.")
@click.option("-o", "--output_file", required=True,
              type=click.Path(dir_okay=False, path_type=pathlib.Path, exists=True),
              help="Path to the database to verify.")
def verify_csv(
    csv_file: pathlib.Path,
    base_dir: Optional[pathlib.Path],
    output_file: pathlib.Path
) -> None:
    if base_dir is None:
        base_dir = csv_file.parent
    db: LMDBFileStorage = LMDBFileStorage(output_file, read_only=True)
    verify_csv_in_db(csv_file, db, base_dir)
    db.close()


@cli.command()
@click.option("-d", "--database", required=True,
              type=click.Path(dir_okay=False, path_type=pathlib.Path, exists=True),
              help="Database whose keys will be printed.")
@click.option("-h", "--hierarchical", is_flag=True,
              help="List files in DB according to directories hierarchy.")
def list_db(
    database: pathlib.Path,
    hierarchical: bool
) -> None:
    """Lists the contents of a file storage."""
    db: LMDBFileStorage = LMDBFileStorage(database, read_only=True)

    if not hierarchical:
        for k in db.get_all_ids():
            print(k)
    else:
        # The db contains filenames as keys, so their parents will always be the dir names.
        ids: list[str] = [str(pathlib.Path(str(k, 'UTF-8')).parent) for k in db.get_all_ids()]
        counts: Counter = Counter(ids)

        dir_graph: nx.DiGraph = nx.DiGraph()
        for k in counts.keys():
            dir_graph.add_edge(str(pathlib.Path(k).parent), k)
            dir_graph.nodes[k]["items_num"] = counts[k]

        top_level_nodes: list[str] = [n for n in dir_graph.nodes if dir_graph.in_degree(n) == 0]
        top_level_nodes = sorted(top_level_nodes)

        for n in top_level_nodes:
            print_dirs_from_graph(dir_graph, n)


def add_csv_to_db(
    csv_file: pathlib.Path,
    db: LMDBFileStorage,
    base_dir: pathlib.Path,
    key_base_dir: Optional[pathlib.Path] = None,
    verbose: bool = True
) -> int:
    """Adds the contents of the file paths included in a CSV file into an LMDB File Storage.

    Paths of the files, relative to the base dir, are utilized as keys into the storage.
    Thus, the maximum allowed path length is 511 bytes.

    The contents of nested CSV files are recursively added into the LMDB File Storage.
    In that case, keys represent the file structure relative to the base dir.

    :param csv_file: Path to a CSV file describing a dataset.
    :param db: An instance of LMDB File Storage, where files will be added.
    :param base_dir: Directory where paths included into the CSV file are relative to.
    :param key_base_dir: Directory where paths encoded into the keys of the LMDB File
        Storage will be relative to. It should be either the same or an upper directory
        compared to base dir. When this argument is omitted, the value of base dir
        is used.
    :param verbose: When set to False, progress messages will not be printed.
    """
    entries: list[dict[str, str]] = read_csv_file(csv_file, verbose=verbose)

    if key_base_dir is None:
        key_base_dir = base_dir

    if verbose:
        pbar = tqdm.tqdm(entries, desc="Writing CSV data to database", unit="file")
    else:
        pbar = entries

    files_written: int = 0
    for e in pbar:
        # Generate key-path pairs for each path in the CSV.
        files_to_write: dict[str, pathlib.Path] = find_files(
            list(e.values()),
            base_dir,
            key_base_dir
        )

        files_written += write_files_to_db(files_to_write, db)

        # Recursively add the contents of the encountered CSV files.
        for p in files_to_write.values():
            if p.suffix == ".csv":
                files_written += add_csv_to_db(
                    p, db, p.parent, key_base_dir=key_base_dir, verbose=False
                )

        if verbose:
            pbar.set_postfix({"Files Written": files_written})

    return files_written


def verify_csv_in_db(
    csv_file: pathlib.Path,
    db: LMDBFileStorage,
    base_dir: pathlib.Path,
    key_base_dir: Optional[pathlib.Path] = None,
    verbose: bool = True
) -> int:
    entries: list[dict[str, str]] = read_csv_file(csv_file, verbose=verbose)

    if key_base_dir is None:
        key_base_dir = base_dir

    if verbose:
        pbar = tqdm.tqdm(entries, desc="Verifying CSV data in database", unit="file")
    else:
        pbar = entries

    files_verified: int = 0
    for e in pbar:
        # Generate key-path pairs for each path in the CSV.
        files: dict[str, pathlib.Path] = find_files(
            list(e.values()),
            base_dir,
            key_base_dir
        )

        files_verified += verify_files_in_db(files, db)

        # Recursively verify the contents of the encountered CSV files.
        for p in files.values():
            if p.suffix == ".csv":
                files_verified += verify_csv_in_db(
                    p, db, p.parent, key_base_dir=key_base_dir, verbose=False
                )

        if verbose:
            pbar.set_postfix({"Files Verified": files_verified})

    return files_verified


def find_files(
    candidates: list[str],
    base_dir: pathlib.Path,
    key_base_dir: pathlib.Path
) -> dict[str, pathlib.Path]:
    files: dict[str, pathlib.Path] = {}
    for c in candidates:
        p: pathlib.Path = base_dir / c
        key: str = str(p.relative_to(key_base_dir))
        if p.exists() and p.is_file():
            files[key] = p
    return files


def write_files_to_db(files: dict[str, pathlib.Path], db: LMDBFileStorage) -> int:
    for k, p in files.items():
        data: bytes = read_raw_file(p)
        db.write_file(k, data)
    return len(files)


def verify_files_in_db(files: dict[str, pathlib.Path], db: LMDBFileStorage) -> int:
    verified: int = 0
    for k, p in files.items():
        # Calculate md5 hash of the file in csv.
        with p.open("rb") as f:
            csv_file_hash: str = md5(f)
        # Calculate md5 hash of the file in db.
        db_file: io.BytesIO = db.open_file(k, mode="b")
        db_file_hash: str = md5(db_file)
        if csv_file_hash == db_file_hash:
            verified += 1
        else:
            logging.error(f"File in DB not matching file in CSV: {str(p)}")
    return verified


def read_csv_file(csv_file: pathlib.Path, verbose: bool = True) -> list[dict[str, str]]:
    # Read the whole csv file.
    if verbose:
        logging.info(f"READING CSV: {str(csv_file)}")

    entries: list[dict[str, str]] = []
    with csv_file.open() as f:
        reader: csv.DictReader = csv.DictReader(f, delimiter=",")
        if verbose:
            pbar = tqdm.tqdm(reader, desc="Reading CSV entries", unit="entry")
        else:
            pbar = reader
        for row in pbar:
            entries.append(row)

    if verbose:
        logging.info(f"TOTAL ENTRIES: {len(entries)}")

    return entries


def read_raw_file(p: pathlib.Path) -> bytes:
    with p.open("rb") as f:
        data: bytes = f.read()
    return data


def md5(stream) -> str:
    """Calculates md5 hash of a file-like stream."""
    hash_md5 = hashlib.md5()
    for chunk in iter(lambda: stream.read(4096), b""):
        hash_md5.update(chunk)
    return hash_md5.hexdigest()


def print_dirs_from_graph(g: nx.DiGraph, n: str, depth: int = 0) -> None:
    if depth > 0:
        init_text: str = "    " * (depth - 1) + "|.. "
    else:
        init_text: str = ""
    text: str = f"{init_text}{n}"
    print(text)

    for s in sorted(g.successors(n)):
        print_dirs_from_graph(g, s, depth+1)

    if "items_num" in g.nodes[n]:
        init_text = "    " * (depth+1)
        text = f"{init_text}({g.nodes[n]['items_num']} files)"
        print(text)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    cli()
