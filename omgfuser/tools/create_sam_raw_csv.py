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

from pathlib import Path

from omgfuser.utils import write_csv_file
from omgfuser.data.filestorage import read_csv_file

csv_p = Path("/fssd2/user-data/dkarageo/CM-JPEG-RAISE/bcmc_COCO_list.csv")
entries = read_csv_file(csv_p)

for e in entries:
    p = Path(e["image"])
    sam_raw = f"CM-JPEG-RAISE/Predictions/SAM/{p.stem}/segmentation_instances.csv"
    e["sam_raw"] = sam_raw
    e["image"] = str(Path("CM-JPEG-RAISE") / e["image"])
    e["mask"] = str(Path("CM-JPEG-RAISE") / e["mask"])

write_csv_file(entries, Path("/fssd2/user-data/dkarageo/cmjpegraise.csv"))
