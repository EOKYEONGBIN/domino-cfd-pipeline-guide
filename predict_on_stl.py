# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This code shows how to use a trained DoMINO model, with it's corresponding
preprocessing pipeline, to infer values on and around an STL mesh file.

This script uses the meshes from the DrivaerML dataset, however, the logic
is largely the same.  As an overview:
- Load the model
- Set up the preprocessor
- Loop over meshes
- In each mesh, sample random points on the surface, volume, or both
- Preprocess the points and run them through the model
- Process the STL mesh centers, too
- Collect the results and return
- Save the results to file.
"""

import os
import re
import time
from typing import Literal, Any

import hydra
from hydra.utils import to_absolute_path
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pyvista as pv
import torch

# This will set up the cupy-ecosystem and pytorch to share memory pools
from physicsnemo.utils.memory import unified_gpu_memory

import torchinfo
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

from physicsnemo.datapipes.cae.domino_datapipe import (
    DoMINODataPipe,
    create_domino_dataset,
)


from physicsnemo.models.domino.model import DoMINO
# Not re-exported from physicsnemo.models.domino.utils's __init__.py in this
# installed version (2.2.2), even though it's defined there -- import from
# the actual submodule directly.
from physicsnemo.models.domino.utils.utils import sample_points_on_mesh, unnormalize

from utils import (
    ScalingFactors,
    get_keys_to_read,
    coordinate_distributed_environment,
    load_scaling_factors,
)

# This is included for GPU memory tracking:
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import time


# Initialize NVML
nvmlInit()


from physicsnemo.utils.profiling import profile, Profiler


from loss import compute_loss_dict
from utils import get_num_vars


def reject_interior_volume_points(
    preprocessed_data: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Reject volume points that are inside the STL mesh.
    """
    ######################################################
    # Use the sign of the volume SDF to filter out points
    # That are inside the STL mesh
    ######################################################
    sdf_nodes = preprocessed_data["sdf_nodes"]
    # The sfd_nodes tensor typically has shape (n_vol_points, 1)
    valid_volume_idx = sdf_nodes > 0
    # So remove it if it's there:
    valid_volume_idx = valid_volume_idx.squeeze(-1)
    # Apply this selection to all the volume points:
    for key in [
        "volume_mesh_centers",
        "sdf_nodes",
        "pos_volume_closest",
        "pos_volume_center_of_mass",
    ]:
        preprocessed_data[key] = preprocessed_data[key][valid_volume_idx]

    return preprocessed_data


def sample_volume_points(
    c_min: torch.Tensor,
    c_max: torch.Tensor,
    n_points: int,
    device: torch.device,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Generate a set of random points interior to the specified bounding box.

    Args:
        c_min: The minimum coordinate of the bounding box.
        c_max: The maximum coordinate of the bounding box.
        n_points: The number of points to sample.
        device: The device to sample the points on.
        eps: The small edge factor to shift away from the lower bound.
    """
    # We use a small edge factor to shift away from the lower bound,
    # which can, in some cases, be exactly on the border.
    uniform_points = (
        torch.rand(n_points, 3, device=device, dtype=torch.float32) * (1 - 2 * eps)
        + eps
    )
    sampled_volume_points = (c_max - c_min) * uniform_points + c_min
    return sampled_volume_points


def sample_volume_grid_points(
    c_min: torch.Tensor,
    c_max: torch.Tensor,
    dims: tuple[int, int, int],
    device: torch.device,
    eps: float = 1e-4,
):
    """
    Build a regular 3D grid of query points, strictly inside [c_min, c_max]
    (process_volume's sample_in_bbox check uses strict >/<, so touching the
    exact boundary would get those points silently dropped).

    Returns (points shaped (nx*ny*nz, 3), origin, spacing) where `points` is
    ordered flat_idx = i + j*nx + k*nx*ny -- i.e. i (x) fastest, matching
    vtkImageData's point ordering, so the caller can write these results
    straight into a pv.ImageData without any further reshuffling.
    """
    nx, ny, nz = dims
    xs = torch.linspace(c_min[0] + eps, c_max[0] - eps, nx, device=device)
    ys = torch.linspace(c_min[1] + eps, c_max[1] - eps, ny, device=device)
    zs = torch.linspace(c_min[2] + eps, c_max[2] - eps, nz, device=device)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    points = torch.stack([gx, gy, gz], dim=-1)  # (nx, ny, nz, 3)
    points = points.permute(2, 1, 0, 3).reshape(-1, 3)  # x fastest when flattened
    origin = torch.stack([xs[0], ys[0], zs[0]])
    spacing = torch.stack([xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]])
    return points, origin, spacing


def inference_volume_grid_on_single_stl(
    stl_coordinates: torch.Tensor,
    stl_faces: torch.Tensor,
    global_params_values: torch.Tensor,
    global_params_reference: torch.Tensor,
    model: DoMINO,
    datapipe: DoMINODataPipe,
    dims: tuple[int, int, int],
    grid_bbox_min: torch.Tensor | None = None,
    grid_bbox_max: torch.Tensor | None = None,
):
    """
    Query the volume model on a REGULAR grid instead of scattered random
    points. Kit-CAE's Streamlines operator needs real cell connectivity to
    advect through (confirmed by direct testing: a scattered point cloud
    fails with "AttributeError: ... build_element_locator", even with
    DatasetVoxelizationAPI/DatasetVoronoiPointCloudAPI applied) -- a
    vtkImageData (regular grid) has that connectivity implicitly, from its
    dims/origin/spacing alone.

    Each forward pass still processes a fixed-size batch
    (datapipe.config.volume_points_sample points, with kNN neighbors
    computed only within that batch -- an architectural constraint of the
    trained model, not something we can bypass), and process_data() shuffles
    point order within each batch and drops any point the bounding-box/SDF
    checks reject. Rather than track that internal reordering, each
    surviving point's returned (unnormalized) coordinate is rounded back to
    a grid index -- since our query points sit exactly on the analytic grid
    to begin with, this recovers the right raster slot regardless of how a
    batch got shuffled or thinned. Grid points that never get a value
    written this way (inside the solid, or thinned by boundary rounding)
    are left at zero, which is a reasonable default for CFD-obstacle
    masking here.

    model_type == "combined" for the trained model, so process_data() always
    processes a surface branch too (it isn't optional even though we only
    want the volume output here) -- a throwaway random surface sample is
    supplied just to satisfy that; its (unused) output is discarded.

    grid_bbox_min/max optionally restrict WHERE the grid points are placed
    (e.g. a tight box around the STL instead of the full dataset-wide
    domain, for finer spacing near the body without more total points).
    They do NOT affect model normalization -- that always uses the fixed
    datapipe.config.bounding_box_dims the model was trained with.
    """
    device = stl_coordinates.device
    batch_size = datapipe.config.volume_points_sample
    nx, ny, nz = dims
    total_points = nx * ny * nz
    if total_points % batch_size != 0:
        raise ValueError(
            f"grid point count {total_points} must be a multiple of "
            f"volume_points_sample ({batch_size})"
        )

    triangle_vertices = stl_coordinates[stl_faces.reshape((-1, 3))]
    stl_centers = triangle_vertices.mean(dim=1)
    d1 = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    d2 = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    # AUDIT FIX (2026-09-29): torch.linalg.cross(d1, d2) with d1=v1-v0,
    # d2=v2-v0 follows the STL FILE's own vertex winding order, which turned
    # out to be the OPPOSITE convention from the CFD boundary mesh's face
    # normals (confirmed via ablation: mean dot product between the two was
    # -0.9995 across 99.99% of run_10's faces -- i.e. a near-universal sign
    # flip, not a smoothness/resolution issue). The model is highly
    # sensitive to normal DIRECTION as an input feature -- this sign flip
    # alone was the dominant cause of the STL-query R^2 collapse (-0.31 vs
    # 0.996 when queried at the true boundary-mesh points), confirmed by
    # negating just the normal (R^2 -0.31 -> 0.76 on run_10). Swapping the
    # cross-product operand order negates the result to match the CFD mesh's
    # convention.
    stl_mesh_normals = torch.linalg.cross(d2, d1, dim=1)
    normals_norm = torch.linalg.norm(stl_mesh_normals, dim=1)
    stl_mesh_normals = stl_mesh_normals / normals_norm.unsqueeze(1)
    stl_areas = 0.5 * normals_norm

    # c_min/c_max (the model's fixed, dataset-wide domain) are used below
    # ONLY to un-normalize the model's output coordinates back to real
    # units -- that must stay the same box the model was trained with.
    # grid_min/grid_max control WHERE we choose to query within that domain,
    # and can be a tighter, near-body box (grid_bbox_min/max) to get finer
    # spacing without changing how coordinates are normalized for the model.
    c_max = datapipe.config.bounding_box_dims[0]
    c_min = datapipe.config.bounding_box_dims[1]
    grid_min = grid_bbox_min if grid_bbox_min is not None else c_min
    grid_max = grid_bbox_max if grid_bbox_max is not None else c_max
    # Clamp to the training domain: process_volume() (physicsnemo's own code)
    # rejects any query point outside [c_min, c_max] and then demands EXACTLY
    # volume_points_sample survivors per chunk -- since we feed it exactly
    # that many points per chunk with no surplus, a near-body box that pokes
    # outside the domain (e.g. padding below a low-ride-height body's floor)
    # starves a chunk below the required count and crashes with "Volume mesh
    # has fewer points than requested sample size".
    grid_min = torch.maximum(grid_min, c_min)
    grid_max = torch.minimum(grid_max, c_max)
    grid_points, origin, spacing = sample_volume_grid_points(grid_min, grid_max, dims, device)

    output_flat = None  # allocated once we know the model's output width
    for start in range(0, total_points, batch_size):
        chunk = grid_points[start : start + batch_size]

        sampled_points, sampled_faces, sampled_areas, sampled_normals = (
            sample_points_on_mesh(
                stl_coordinates,
                stl_faces,
                batch_size,
                mesh_normals=stl_mesh_normals,
                mesh_areas=stl_areas,
            )
        )
        inference_dict = {
            "stl_coordinates": stl_coordinates,
            "stl_faces": stl_faces,
            "stl_centers": stl_centers,
            "stl_areas": stl_areas,
            "global_params_values": global_params_values,
            "global_params_reference": global_params_reference,
            "surface_mesh_centers": sampled_points,
            "surface_normals": sampled_normals,
            "surface_areas": sampled_areas,
            "surface_faces": sampled_faces,
            "volume_mesh_centers": chunk,
        }

        preprocessed_data = datapipe.process_data(inference_dict)
        preprocessed_data = reject_interior_volume_points(preprocessed_data)
        if preprocessed_data["volume_mesh_centers"].shape[0] == 0:
            continue

        chunk_coords_norm = preprocessed_data["volume_mesh_centers"].clone()
        batched = {k: v.unsqueeze(0) for k, v in preprocessed_data.items()}
        with torch.no_grad():
            output_vol, _ = model(batched)
        output_vol, _ = datapipe.unscale_model_outputs(output_vol, None)
        output_vol = output_vol[0]

        if datapipe.config.normalize_coordinates:
            chunk_coords = unnormalize(chunk_coords_norm, c_max, c_min)
        else:
            chunk_coords = chunk_coords_norm

        if output_flat is None:
            output_flat = torch.zeros(
                (total_points, output_vol.shape[-1]),
                device=device,
                dtype=output_vol.dtype,
            )

        idx = torch.round((chunk_coords - origin) / spacing).long()
        idx = idx.clamp(
            min=torch.zeros(3, dtype=torch.long, device=device),
            max=torch.tensor([nx - 1, ny - 1, nz - 1], device=device),
        )
        flat_idx = idx[:, 0] + idx[:, 1] * nx + idx[:, 2] * nx * ny
        output_flat[flat_idx] = output_vol

    return output_flat, origin, spacing


def save_volume_grid_prediction(
    output_flat: torch.Tensor,
    origin: torch.Tensor,
    spacing: torch.Tensor,
    dims: tuple[int, int, int],
    volume_solution_cfg: dict,
    output_path: str,
) -> None:
    """
    Write the grid-queried volume fields as a vtkImageData (.vti) -- unlike
    save_volume_prediction's scattered point cloud, this has real cell
    connectivity, which Kit-CAE's Streamlines operator needs to advect
    through.
    """
    nx, ny, nz = dims
    image = pv.ImageData(
        dimensions=(nx, ny, nz),
        origin=tuple(origin.detach().cpu().numpy().tolist()),
        spacing=tuple(spacing.detach().cpu().numpy().tolist()),
    )
    results_np = output_flat.detach().cpu().numpy()
    offset = 0
    for name, kind in volume_solution_cfg.items():
        dim = 3 if kind == "vector" else 1
        image.point_data[name] = results_np[:, offset : offset + dim]
        offset += dim

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)


def inference_surface_mesh_on_single_stl(
    stl_coordinates: torch.Tensor,
    stl_faces: torch.Tensor,
    global_params_values: torch.Tensor,
    global_params_reference: torch.Tensor,
    model: DoMINO,
    datapipe: DoMINODataPipe,
):
    """
    Evaluate the model at EVERY face of a 1-level subdivided copy of the STL
    (not a random 8192-point subsample), chunked through the model in batches
    of datapipe.config.surface_points_sample. Returns
    (subdivided_vertices, subdivided_faces, per_face_output) so the output
    has real triangle connectivity (CreateCaeVizFaces), matching how the ground-truth
    boundary_*.vtp examples work, instead of being a disconnected point
    cloud limited to one subsampled batch (CreateCaeVizPoints; see
    inference_on_single_stl/save_surface_prediction).

    process_surface() shuffles point order *within* each batch (and drops
    any zero-area/degenerate faces) with no public way to recover the
    shuffle indices, so each chunk's returned (unnormalized) coordinates
    are matched back to that SAME chunk's own known candidate face centers
    via nearest-neighbor -- a small, per-chunk search, since points are
    never shuffled *across* chunks, only within one. The last chunk is
    padded by repeating a few of its own indices (process_surface requires
    at least a full batch); the padding's predictions just harmlessly
    overwrite an already-computed face with an equivalent value picked up
    by the same nearest-neighbor match, so no separate discard step is
    needed.

    model_type == "combined" for the trained model, so process_data()
    always processes a volume branch too -- a throwaway random volume
    sample is supplied just to satisfy that; its (unused) output is
    discarded.
    """
    device = stl_coordinates.device
    batch_size = datapipe.config.surface_points_sample

    # Original-resolution STL geometry -- unchanged, still fed to
    # process_data() as "stl_coordinates"/"stl_faces"/"stl_centers"/
    # "stl_areas" for the geometry encoder (SDF etc.), which should reflect
    # the STL's true triangulation regardless of what resolution the surface
    # solution is *queried* at below.
    triangle_vertices = stl_coordinates[stl_faces.reshape((-1, 3))]
    stl_centers = triangle_vertices.mean(dim=1)
    d1 = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    d2 = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    # AUDIT FIX (2026-09-29): torch.linalg.cross(d1, d2) with d1=v1-v0,
    # d2=v2-v0 follows the STL FILE's own vertex winding order, which turned
    # out to be the OPPOSITE convention from the CFD boundary mesh's face
    # normals (confirmed via ablation: mean dot product between the two was
    # -0.9995 across 99.99% of run_10's faces -- i.e. a near-universal sign
    # flip, not a smoothness/resolution issue). The model is highly
    # sensitive to normal DIRECTION as an input feature -- this sign flip
    # alone was the dominant cause of the STL-query R^2 collapse (-0.31 vs
    # 0.996 when queried at the true boundary-mesh points), confirmed by
    # negating just the normal (R^2 -0.31 -> 0.76 on run_10). Swapping the
    # cross-product operand order negates the result to match the CFD mesh's
    # convention.
    stl_mesh_normals = torch.linalg.cross(d2, d1, dim=1)
    normals_norm = torch.linalg.norm(stl_mesh_normals, dim=1)
    stl_areas = 0.5 * normals_norm
    stl_mesh_normals = stl_mesh_normals / normals_norm.clamp(min=1e-12).unsqueeze(1)

    n_orig_faces = stl_centers.shape[0]

    # AUDIT FIX (2026-09-29): even with the normal SIGN corrected above, the
    # STL's own (much coarser) triangulation still gives each face a larger,
    # less locally-accurate area/normal than the CFD boundary mesh DoMINO was
    # trained on (194,940 STL faces vs. run_10's 813,548 CFD cells for
    # AhmedML) -- confirmed this alone still capped R^2 at ~0.76. Querying
    # the model at a SUBDIVIDED (~4x finer) version of the STL instead closes
    # most of that gap (R^2 -> ~0.99, vs. a 0.994 ceiling from substituting
    # the CFD mesh's own normals/areas directly) -- purely geometric, no
    # ground-truth data used, so it applies to a genuinely new/unseen
    # geometry too. 1 level of linear subdivision splits each triangle into
    # 4, landing very close to the CFD mesh's resolution for AhmedML.
    #
    # The subdivided mesh itself is returned and saved (2026-10-06): it used
    # to be averaged back onto the original STL faces, which threw away the
    # finer result and showed visibly coarser, stair-stepped color bands than
    # the CFD boundary mesh in Kit-CAE (run_10 front: ~3.1mm STL faces vs
    # ~1.2mm CFD cells).
    faces_np = stl_faces.reshape((-1, 3)).detach().cpu().numpy().astype(np.int64)
    faces_flat = np.hstack(
        [np.full((n_orig_faces, 1), 3, dtype=np.int64), faces_np]
    ).reshape(-1)
    subdiv_mesh = pv.PolyData(stl_coordinates.detach().cpu().numpy(), faces_flat)
    subdiv_mesh = subdiv_mesh.subdivide(1, subfilter="linear")

    sub_verts = torch.from_numpy(subdiv_mesh.points).to(device=device, dtype=stl_coordinates.dtype)
    sub_faces_idx = torch.from_numpy(
        subdiv_mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
    ).to(device)
    sub_triangle_vertices = sub_verts[sub_faces_idx]
    query_centers = sub_triangle_vertices.mean(dim=1)
    qd1 = sub_triangle_vertices[:, 1] - sub_triangle_vertices[:, 0]
    qd2 = sub_triangle_vertices[:, 2] - sub_triangle_vertices[:, 0]
    query_normals_raw = torch.linalg.cross(qd2, qd1, dim=1)
    query_normals_norm = torch.linalg.norm(query_normals_raw, dim=1)
    query_areas = 0.5 * query_normals_norm
    query_normals = query_normals_raw / query_normals_norm.clamp(min=1e-12).unsqueeze(1)

    num_faces = query_centers.shape[0]  # subdivided count, not n_orig_faces
    output_full = None  # allocated once the model's output width is known

    # Chunk over only the non-degenerate (subdivided) faces. process_surface()
    # internally drops any face with surface_sizes <= 0, so if a chunk we
    # submit already contains degenerate faces, the count it actually samples
    # from can fall below batch_size and it raises "Surface mesh has fewer
    # points than requested sample size" -- confirmed on a heavily
    # hand-edited STL with clustered degenerate triangles. Restricting the
    # candidate pool up front means every chunk (including the padded last
    # one, which only ever repeats already-valid indices) is guaranteed to
    # have exactly batch_size valid faces. Degenerate faces are left at the
    # default zero in the final output, same as the (rare) unmatched-face
    # case on ordinary meshes.
    valid_face_idx = torch.nonzero(query_areas > 0, as_tuple=True)[0]
    num_valid = valid_face_idx.shape[0]

    needs_volume_branch = datapipe.model_type in ("volume", "combined")
    if needs_volume_branch:
        c_max = datapipe.config.bounding_box_dims[0]
        c_min = datapipe.config.bounding_box_dims[1]

    for start in range(0, num_valid, batch_size):
        end = min(start + batch_size, num_valid)
        chunk_valid_pos = torch.arange(start, end, device=device)
        if chunk_valid_pos.shape[0] < batch_size:
            pad = batch_size - chunk_valid_pos.shape[0]
            # A single slice of the chunk isn't necessarily enough padding
            # material -- when there are few enough total faces that the
            # last chunk's shortfall exceeds its own size (confirmed: 3012
            # valid faces short by 5180 on a heavily edited, lower-face-count
            # STL), tile it as many times as needed before trimming.
            repeats = -(-pad // chunk_valid_pos.shape[0])  # ceil division
            chunk_valid_pos = torch.cat(
                [chunk_valid_pos, chunk_valid_pos.repeat(repeats)[:pad]]
            )
        chunk_indices = valid_face_idx[chunk_valid_pos]

        chunk_coords = query_centers[chunk_indices]

        inference_dict = {
            "stl_coordinates": stl_coordinates,
            "stl_faces": stl_faces,
            "stl_centers": stl_centers,
            "stl_areas": stl_areas,
            "global_params_values": global_params_values,
            "global_params_reference": global_params_reference,
            "surface_mesh_centers": chunk_coords,
            "surface_normals": query_normals[chunk_indices],
            "surface_areas": query_areas[chunk_indices],
        }
        if needs_volume_branch:
            inference_dict["volume_mesh_centers"] = sample_volume_points(
                c_min, c_max, batch_size, device
            )

        preprocessed_data = datapipe.process_data(inference_dict)
        if needs_volume_branch:
            preprocessed_data = reject_interior_volume_points(preprocessed_data)

        chunk_coords_norm_out = preprocessed_data["surface_mesh_centers"].clone()
        batched = {k: v.unsqueeze(0) for k, v in preprocessed_data.items()}
        with torch.no_grad():
            _, output_surf = model(batched)
        _, output_surf = datapipe.unscale_model_outputs(None, output_surf)
        output_surf = output_surf[0]

        if datapipe.config.normalize_coordinates:
            s_max = datapipe.config.bounding_box_dims_surf[0]
            s_min = datapipe.config.bounding_box_dims_surf[1]
            chunk_coords_out = unnormalize(chunk_coords_norm_out, s_max, s_min)
        else:
            chunk_coords_out = chunk_coords_norm_out

        if output_full is None:
            output_full = torch.zeros(
                (num_faces, output_surf.shape[-1]), device=device, dtype=output_surf.dtype
            )

        dists = torch.cdist(chunk_coords_out, chunk_coords)
        nearest = dists.argmin(dim=1)
        matched_face_idx = chunk_indices[nearest]
        output_full[matched_face_idx] = output_surf

    return sub_verts, sub_faces_idx, output_full


def save_surface_mesh_prediction(
    verts: torch.Tensor,
    faces: torch.Tensor,
    output_full: torch.Tensor,
    surface_solution_cfg: dict,
    output_path: str,
) -> None:
    """
    Write predicted surface fields on the (subdivided) query mesh -- real
    triangle connectivity, so Kit-CAE's Faces operator can be used directly
    instead of Points. Each field is stored as cell data under its own name
    (e.g. "pMean"), one value per face, the same layout as the CFD
    boundary_*.vtp files.
    """
    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.reshape((-1, 3)).detach().cpu().numpy().astype(np.int64)
    n_faces = faces_np.shape[0]
    faces_flat = np.hstack(
        [np.full((n_faces, 1), 3, dtype=np.int64), faces_np]
    ).reshape(-1)
    mesh = pv.PolyData(verts_np, faces_flat)

    results_np = output_full.detach().cpu().numpy()
    offset = 0
    for name, kind in surface_solution_cfg.items():
        dim = 3 if kind == "vector" else 1
        mesh.cell_data[name] = results_np[:, offset : offset + dim]
        offset += dim

    # Cp = (p - p_ref) / (0.5 * rho * U_inf^2); AhmedML's global_parameters
    # (inlet_velocity=1.0, air_density=1.0) is OpenFOAM's non-dimensional
    # kinematic-pressure convention (p already means p/rho), so this reduces
    # to Cp = 2 * (pMean - p_ref). p_ref=0 verified directly against
    # CAE_Examples/AhmedML/run_1/boundary_1.vtp's own pMean vs.
    # static(p)_coeffMean fields (min/max matched 2*pMean almost exactly).
    # Not a model output -- a plain linear derivation from the already
    # -predicted pMean, so no retraining/re-inference is needed for this.
    if "pMean" in mesh.cell_data:
        mesh.cell_data["static(p)_coeffMean"] = 2.0 * mesh.cell_data["pMean"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Testing binary (default) instead of ASCII here -- suspect ASCII +
    # cell-data (as opposed to point-data) association is what's actually
    # breaking Kit-CAE's Faces coloring, since Streamlines' point-data VTI
    # works fine while this cell-data VTP renders blank/white.
    mesh.save(output_path)
    _strip_empty_topology_elements(output_path)


def inference_on_single_stl(
    stl_coordinates: torch.Tensor,
    stl_faces: torch.Tensor,
    global_params_values: torch.Tensor,
    global_params_reference: torch.Tensor,
    model: DoMINO,
    datapipe: DoMINODataPipe,
    batch_size: int,
    total_points: int,
    gpu_handle: int | None = None,
    logger: PythonLogger | None = None,
):
    """
    Perform model inference on a single STL mesh.

    This function will take the input mesh + faces and
    then sample the surface and volume to produce the model outputs
    at `total_points` locations in batches of `batch_size`.



    Args:
        stl_coordinates: The coordinates of the STL mesh.
        stl_faces: The faces of the STL mesh.
        global_params_values: The values of the global parameters.
        global_params_reference: The reference values of the global parameters.
        model: The model to use for inference.
        datapipe: The datapipe to use for preprocessing.
        batch_size: The batch size to use for inference.
        total_points: The total number of points to process.
        gpu_handle: The GPU handle to use for inference.
        logger: The logger to use for logging.
    """
    device = stl_coordinates.device
    batch_start_time = time.perf_counter()
    ######################################################
    # The IO only reads in "stl_faces" and "stl_coordinates".
    # "stl_areas" and "stl_centers" would be computed by
    # pyvista on CPU - instead, we do it on the GPU
    # right here.
    ######################################################

    # Center is a mean of the 3 vertices. triangle_vertices has shape
    # (F, 3_verts, 3_coords) -- must average over the vertex axis (dim=1),
    # not the coordinate axis (dim=-1, a bug that silently produced
    # (x+y+z)/3-style garbage "centers" with the right shape but nonsense
    # values, corrupting both the output point-cloud coordinates and the
    # center-of-mass model input feature computed from these).
    triangle_vertices = stl_coordinates[stl_faces.reshape((-1, 3))]
    stl_centers = triangle_vertices.mean(dim=1)
    ######################################################
    # Area we compute from the cross product of two sides:
    ######################################################
    d1 = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    d2 = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    # AUDIT FIX (2026-09-29): torch.linalg.cross(d1, d2) with d1=v1-v0,
    # d2=v2-v0 follows the STL FILE's own vertex winding order, which turned
    # out to be the OPPOSITE convention from the CFD boundary mesh's face
    # normals (confirmed via ablation: mean dot product between the two was
    # -0.9995 across 99.99% of run_10's faces -- i.e. a near-universal sign
    # flip, not a smoothness/resolution issue). The model is highly
    # sensitive to normal DIRECTION as an input feature -- this sign flip
    # alone was the dominant cause of the STL-query R^2 collapse (-0.31 vs
    # 0.996 when queried at the true boundary-mesh points), confirmed by
    # negating just the normal (R^2 -0.31 -> 0.76 on run_10). Swapping the
    # cross-product operand order negates the result to match the CFD mesh's
    # convention.
    stl_mesh_normals = torch.linalg.cross(d2, d1, dim=1)
    normals_norm = torch.linalg.norm(stl_mesh_normals, dim=1)
    stl_mesh_normals = stl_mesh_normals / normals_norm.unsqueeze(1)
    stl_areas = 0.5 * normals_norm

    ######################################################
    # For computing the points, we take those stl objects,
    # sample in chunks of `batch_size` until we've
    # accumulated `total_points` predictions.
    ######################################################

    batch_output_dict = {}
    N = 2
    total_points_processed = 0

    # Use these lists to build up the output tensors:
    surface_results = []
    volume_results = []
    volume_coords_list = []

    while total_points_processed < total_points:
        inner_loop_start_time = time.perf_counter()

        ######################################################
        # Create the dictionary as the preprocessing expects:
        ######################################################
        inference_dict = {
            "stl_coordinates": stl_coordinates,
            "stl_faces": stl_faces,
            "stl_centers": stl_centers,
            "stl_areas": stl_areas,
            "global_params_values": global_params_values,
            "global_params_reference": global_params_reference,
        }

        # If the surface data is part of the model, sample the surface:

        if datapipe.model_type == "surface" or datapipe.model_type == "combined":
            ######################################################
            # This function will sample points on the STL surface
            ######################################################
            sampled_points, sampled_faces, sampled_areas, sampled_normals = (
                sample_points_on_mesh(
                    stl_coordinates,
                    stl_faces,
                    batch_size,
                    mesh_normals=stl_mesh_normals,
                    mesh_areas=stl_areas,
                )
            )

            inference_dict["surface_mesh_centers"] = sampled_points
            inference_dict["surface_normals"] = sampled_normals
            inference_dict["surface_areas"] = sampled_areas
            inference_dict["surface_faces"] = sampled_faces

        # If the volume data is part of the model, sample the volume:
        if datapipe.model_type == "volume" or datapipe.model_type == "combined":
            ######################################################
            # Build up volume points too with uniform sampling
            ######################################################
            c_min = datapipe.config.bounding_box_dims[1]
            c_max = datapipe.config.bounding_box_dims[0]
            inference_dict["volume_mesh_centers"] = sample_volume_points(
                c_min,
                c_max,
                batch_size,
                device,
            )

        ######################################################
        # Pre-process the data with the datapipe:
        ######################################################
        preprocessed_data = datapipe.process_data(inference_dict)

        if datapipe.model_type == "volume" or datapipe.model_type == "combined":
            preprocessed_data = reject_interior_volume_points(preprocessed_data)
            # Captured post-rejection so it stays row-aligned with output_vol
            # (still normalized to [-1, 1] here -- unnormalized once, after
            # the loop, alongside the rest of the accumulated batches).
            volume_coords_list.append(preprocessed_data["volume_mesh_centers"].clone())

        ######################################################
        # Add a batch dimension to the data_dict
        # (normally this is added in __getitem__ of the datapipe)
        ######################################################
        preprocessed_data = {k: v.unsqueeze(0) for k, v in preprocessed_data.items()}

        ######################################################
        # Forward pass through the model:
        ######################################################
        with torch.no_grad():
            output_vol, output_surf = model(preprocessed_data)

        ######################################################
        # unnormalize the outputs with the datapipe
        # Whatever settings are configured for normalizing the
        # output fields - even though we don't have ground
        # truth here - are reused to undo that for the predictions
        ######################################################
        output_vol, output_surf = datapipe.unscale_model_outputs(
            output_vol, output_surf
        )

        surface_results.append(output_surf)
        volume_results.append(output_vol)

        total_points_processed += batch_size

        current_loop_time = time.perf_counter()

        logging_string = f"Device {device} processed {total_points_processed} points of {total_points}\n"
        if gpu_handle is not None:
            gpu_info = nvmlDeviceGetMemoryInfo(gpu_handle)
            gpu_memory_used = gpu_info.used / (1024**3)
            logging_string += f"  GPU memory used: {gpu_memory_used:.3f} Gb\n"

        logging_string += f"  Time taken since batch start: {current_loop_time - batch_start_time:.2f} seconds\n"
        logging_string += f"  iteration throughput: {batch_size / (current_loop_time - inner_loop_start_time):.1f} points per second\n"
        logging_string += f"  Batch mean throughput: {total_points_processed / (current_loop_time - batch_start_time):.1f} points per second.\n"

        if logger is not None:
            logger.info(logging_string)
        else:
            print(logging_string)

    ######################################################
    # Here at the end, get the values for the stl centers
    # by updating the previous inference dict
    # Only do this if the surface is part of the computation
    # Comments are shorter here - it's a condensed version
    # of the above logic.
    ######################################################
    if datapipe.model_type == "surface" or datapipe.model_type == "combined":
        inference_dict = {
            "stl_coordinates": stl_coordinates,
            "stl_faces": stl_faces,
            "stl_centers": stl_centers,
            "stl_areas": stl_areas,
            "global_params_values": global_params_values,
            "global_params_reference": global_params_reference,
        }
        inference_dict["surface_mesh_centers"] = stl_centers
        inference_dict["surface_normals"] = stl_mesh_normals
        inference_dict["surface_areas"] = stl_areas
        inference_dict["surface_faces"] = stl_faces

        if datapipe.model_type == "combined" or datapipe.model_type == "volume":
            c_min = datapipe.config.bounding_box_dims[1]
            c_max = datapipe.config.bounding_box_dims[0]
            inference_dict["volume_mesh_centers"] = sample_volume_points(
                c_min,
                c_max,
                stl_centers.shape[0],
                device,
            )

        # Preprocess:
        preprocessed_data = datapipe.process_data(inference_dict)

        # Pull out the invalid volume points again, if needed:
        if datapipe.model_type == "combined" or datapipe.model_type == "volume":
            preprocessed_data = reject_interior_volume_points(preprocessed_data)

        # NOTE: process_data() subsamples "surface_mesh_centers" down to
        # model.surface_points_sample points (e.g. 8192) -- for meshes with
        # more faces than that (typical for AhmedML/DrivAerML), the model
        # output does NOT correspond 1:1 with the original mesh faces. Save
        # off the actual (subsampled) query coordinates here so the caller
        # can build an output point cloud at the coordinates that were
        # actually evaluated, rather than assuming a 1:1 mapping to the
        # original STL faces.
        stl_center_coords = preprocessed_data["surface_mesh_centers"].clone()

        # process_surface() (inside datapipe.process_data()) normalizes
        # "surface_mesh_centers" to [-1, 1] via the *fixed*, dataset-wide
        # data.bounding_box_surface min/max (not the per-STL bounding box) --
        # confirmed in physicsnemo's domino_datapipe.py/models/domino/utils.
        # Undo that here so the output point cloud lands in the same
        # physical (meters) coordinate frame as the original STL, instead of
        # a tiny [-1, 1]-normalized blob offset from the actual geometry.
        if datapipe.config.normalize_coordinates:
            s_max, s_min = datapipe.config.bounding_box_dims_surf
            stl_center_coords = unnormalize(stl_center_coords, s_max, s_min)

        # Run the model forward:
        with torch.no_grad():
            preprocessed_data = {
                k: v.unsqueeze(0) for k, v in preprocessed_data.items()
            }
            _, output_surf = model(preprocessed_data)

        # Unnormalize the outputs:
        _, stl_center_results = datapipe.unscale_model_outputs(None, output_surf)

    else:
        stl_center_results = None
        stl_center_coords = None

    # Stack up the results into one big tensor for surface and volume:
    if len(surface_results) > 0 and all([s is not None for s in surface_results]):
        surface_results = torch.cat(surface_results, dim=1)
    else:
        surface_results = None
    if len(volume_results) > 0 and all([v is not None for v in volume_results]):
        volume_results = torch.cat(volume_results, dim=1)
        volume_coords = torch.cat(volume_coords_list, dim=0)
        # Same [-1, 1] normalization (via the fixed volume bounding_box_dims,
        # not per-STL) as the surface case -- undo it the same way.
        if datapipe.config.normalize_coordinates:
            c_max, c_min = datapipe.config.bounding_box_dims
            volume_coords = unnormalize(volume_coords, c_max, c_min)
    else:
        volume_results = None
        volume_coords = None

    return stl_center_results, stl_center_coords, surface_results, volume_results, volume_coords


def save_surface_prediction(
    stl_center_coords: torch.Tensor,
    stl_center_results: torch.Tensor,
    surface_solution_cfg: dict,
    output_path: str,
) -> None:
    """
    Write the predicted surface fields (pressure, wall shear stress, etc.)
    as a point cloud .vtp that Kit-CAE can import directly.

    NOTE: this is a point cloud, not a full-resolution triangulated surface.
    process_data() subsamples the query points down to
    model.surface_points_sample (e.g. 8192) per forward pass, so for meshes
    with more faces than that (typical for AhmedML/DrivAerML), the model
    output does not correspond 1:1 with the original STL faces -- there is
    no original per-face connectivity to reattach these predictions to.
    stl_center_coords are the actual (subsampled) coordinates the model was
    evaluated at, so a point cloud is the correct-resolution representation
    of what was actually predicted.
    """
    points_np = stl_center_coords.detach().cpu().numpy()
    if points_np.ndim == 3:  # squeeze a leftover batch dim, if present
        points_np = points_np[0]
    mesh = pv.PolyData(points_np)

    results_np = stl_center_results.detach().cpu().numpy()[0]
    offset = 0
    for name, kind in surface_solution_cfg.items():
        dim = 3 if kind == "vector" else 1
        mesh.point_data[name] = results_np[:, offset : offset + dim]
        offset += dim

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # ascii (uncompressed, no zlib header) avoids a separate Kit-CAE VTP
    # parser bug specific to binary/compressed Verts-only polydata ("VTK XML
    # direct compressed target size does not match decoded payload size"),
    # on top of the empty-topology-element bug _strip_empty_topology_elements
    # works around. Larger file, but these prediction point clouds are small.
    mesh.save(output_path, binary=False)
    _strip_empty_topology_elements(output_path)


def save_volume_prediction(
    volume_coords: torch.Tensor,
    volume_results: torch.Tensor,
    volume_solution_cfg: dict,
    output_path: str,
) -> None:
    """
    Write the predicted volume fields (velocity, pressure, turbulent
    viscosity) as a point cloud .vtp -- a scattered sampling of the fluid
    domain around the STL (points inside the solid are already excluded by
    reject_interior_volume_points upstream).

    This is a point cloud, not a structured grid: DoMINO's model queries a
    fixed-size, randomly sampled batch of points per forward pass (kNN
    neighbors are computed within each batch), so there's no simple way to
    query a regular raster grid without arbitrarily chunking it and
    approximating neighbor relationships. Kit-CAE's streamline operator can
    still integrate through this by voxelizing the point cloud first
    (CaeVizDatasetVoxelizationAPI) into a continuous field, the same
    approach already used for the input-shape-independent GB300 example.
    """
    points_np = volume_coords.detach().cpu().numpy()
    if points_np.ndim == 3:  # squeeze a leftover batch dim, if present
        points_np = points_np[0]
    mesh = pv.PolyData(points_np)

    results_np = volume_results.detach().cpu().numpy()[0]
    offset = 0
    for name, kind in volume_solution_cfg.items():
        dim = 3 if kind == "vector" else 1
        mesh.point_data[name] = results_np[:, offset : offset + dim]
        offset += dim

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Same ASCII workaround as save_surface_prediction (see its comment).
    mesh.save(output_path, binary=False)
    _strip_empty_topology_elements(output_path)


def _strip_empty_topology_elements(vtp_path: str) -> None:
    """
    PyVista's VTP writer always emits all four topology elements (Verts,
    Lines, Strips, Polys), leaving whichever ones a given mesh doesn't use
    empty (NumberOfX="0" on the <Piece> tag) but still PRESENT. Kit-CAE's
    VTP importer has a bug where the mere PRESENCE of an empty topology
    element -- regardless of which kind -- breaks it: an empty <Strips>
    outright fails import (confirmed with GB300's structure.vtp, filed as
    https://github.com/NVIDIA-Omniverse/kit-cae/issues/26); empty
    <Verts>/<Lines> on our own Faces-based surface mesh output didn't block
    import but silently broke the "colors" field-selection dropdown's
    reactivity (switching fields stopped updating the render). Only
    elements that are actually empty for THIS file get removed -- e.g. a
    genuine point-cloud file's non-empty <Verts> is left alone. Text-level
    removal is safe here because these files use format="binary"
    (self-contained per-DataArray), not offset-indexed "appended" data.
    """
    with open(vtp_path, "r", encoding="utf-8") as f:
        content = f.read()
    for tag, count_attr in [
        ("Verts", "NumberOfVerts"),
        ("Lines", "NumberOfLines"),
        ("Strips", "NumberOfStrips"),
        ("Polys", "NumberOfPolys"),
    ]:
        if re.search(rf'{count_attr}="0"', content):
            content = re.sub(rf"<{tag}>.*?</{tag}>\s*", "", content, flags=re.DOTALL)
    with open(vtp_path, "w", encoding="utf-8") as f:
        f.write(content)


def inference_epoch(
    dataloader: DoMINODataPipe,
    sampler: DistributedSampler,
    model: DoMINO,
    gpu_handle: int,
    logger: PythonLogger,
    batch_size: int = 24_000,
    total_points: int = 1_024_000,
    surface_solution_cfg: dict | None = None,
    volume_solution_cfg: dict | None = None,
    volume_grid_dims: tuple[int, int, int] | None = None,
    save_path: str | None = None,
):
    ######################################################
    # Inference can run in a distributed way by coordinating
    # the indices for each rank, which the sampler does
    ######################################################

    batch_start_time = time.perf_counter()

    # N.B. - iterating over the dataset directly here.
    # That's because we need to sample on the STL and volume and
    # that means we'll preprocess after that.
    for i_batch, sample_batched in enumerate(dataloader.dataset):
        dataloading_time = time.perf_counter() - batch_start_time

        logger.info(
            f"Batch {i_batch} data loading time: {dataloading_time:.3f} seconds"
        )

        procesing_time_start = time.perf_counter()
        stl_center_results, stl_center_coords, surface_results, volume_results, volume_coords = inference_on_single_stl(
            sample_batched["stl_coordinates"],
            sample_batched["stl_faces"],
            sample_batched["global_params_values"],
            sample_batched["global_params_reference"],
            model,
            dataloader,
            batch_size,
            total_points,
            gpu_handle,
            logger,
        )

        ######################################################
        # Peel off pressure, velocity, nut, shear, etc.
        # Also compute drag, lift forces.
        ######################################################
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO

        procesing_time_end = time.perf_counter()
        logger.info(
            f"Batch {i_batch} GPU processing time: {procesing_time_end - procesing_time_start:.3f} seconds"
        )
        logger.info(
            f"Batch {i_batch} stl points: {stl_center_results.shape[1] if stl_center_results is not None else 0}"
        )

        output_start_time = time.perf_counter()
        ######################################################
        # Save the outputs to file:
        ######################################################
        if surface_solution_cfg is not None and save_path is not None:
            sub_verts, sub_faces, surface_output_full = inference_surface_mesh_on_single_stl(
                sample_batched["stl_coordinates"],
                sample_batched["stl_faces"],
                sample_batched["global_params_values"],
                sample_batched["global_params_reference"],
                model,
                dataloader,
            )
            case_output_path = os.path.join(save_path, f"prediction_{i_batch}.vtp")
            save_surface_mesh_prediction(
                sub_verts,
                sub_faces,
                surface_output_full,
                surface_solution_cfg,
                case_output_path,
            )
            logger.info(f"Batch {i_batch} full-resolution surface mesh prediction saved to {case_output_path}")

        if (
            volume_results is not None
            and volume_solution_cfg is not None
            and save_path is not None
        ):
            case_volume_output_path = os.path.join(save_path, f"prediction_volume_{i_batch}.vtp")
            save_volume_prediction(
                volume_coords,
                volume_results,
                volume_solution_cfg,
                case_volume_output_path,
            )
            logger.info(f"Batch {i_batch} volume prediction saved to {case_volume_output_path}")

        if (
            volume_solution_cfg is not None
            and volume_grid_dims is not None
            and save_path is not None
        ):
            # Query a box just around this STL's own bounds (expanded by 1x
            # its own size per axis) instead of the full dataset-wide domain
            # -- concentrates the same point budget near the body, where
            # local surface features actually show up, instead of spread
            # across the whole (mostly empty) far-field.
            stl_min = sample_batched["stl_coordinates"].min(dim=0).values
            stl_max = sample_batched["stl_coordinates"].max(dim=0).values
            stl_size = stl_max - stl_min
            grid_bbox_min = stl_min - stl_size
            grid_bbox_max = stl_max + stl_size

            grid_output_flat, grid_origin, grid_spacing = inference_volume_grid_on_single_stl(
                sample_batched["stl_coordinates"],
                sample_batched["stl_faces"],
                sample_batched["global_params_values"],
                sample_batched["global_params_reference"],
                model,
                dataloader,
                volume_grid_dims,
                grid_bbox_min=grid_bbox_min,
                grid_bbox_max=grid_bbox_max,
            )
            if grid_output_flat is not None:
                case_volume_grid_output_path = os.path.join(
                    save_path, f"prediction_volume_grid_{i_batch}.vti"
                )
                save_volume_grid_prediction(
                    grid_output_flat,
                    grid_origin,
                    grid_spacing,
                    volume_grid_dims,
                    volume_solution_cfg,
                    case_volume_grid_output_path,
                )
                logger.info(
                    f"Batch {i_batch} volume grid prediction saved to {case_volume_grid_output_path}"
                )
        output_end_time = time.perf_counter()
        logger.info(
            f"Batch {i_batch} output time: {output_end_time - output_start_time:.3f} seconds"
        )

        batch_start_time = time.perf_counter()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    ######################################################
    # initialize distributed manager
    ######################################################
    DistributedManager.initialize()
    dist = DistributedManager()

    # DoMINO supports domain parallel training and inference.  This function helps coordinate
    # how to set that up, if needed.
    domain_mesh, data_mesh, placements = coordinate_distributed_environment(cfg)

    # data_mesh is intentionally None for single-GPU runs (see
    # coordinate_distributed_environment) -- this script never handled that
    # case, unlike train.py, which falls back to dist.world_size/dist.rank.
    if data_mesh is not None:
        data_replica_size = data_mesh.size()
        data_rank = data_mesh.get_local_rank()
    else:
        data_replica_size = dist.world_size
        data_rank = dist.rank

    ######################################################
    # Initialize NVML
    ######################################################
    nvmlInit()
    gpu_handle = nvmlDeviceGetHandleByIndex(dist.device.index)

    ######################################################
    # Initialize logger
    ######################################################

    logger = PythonLogger("Inference")
    logger = RankZeroLoggingWrapper(logger, dist)

    logger.info(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    ######################################################
    # Get scaling factors
    # Likely, you want to reuse the scaling factors from training.
    ######################################################
    vol_factors, surf_factors = load_scaling_factors(cfg)

    ######################################################
    # Configure the model
    ######################################################
    model_type = cfg.model.model_type
    num_vol_vars, num_surf_vars, num_global_features = get_num_vars(cfg, model_type)

    # AUDIT FIX (2026-09-29): Kit-CAE's "Request Prediction" UI lets the user
    # ask for Faces (surface) and/or Streamlines (volume) independently --
    # eval.compute_surface/compute_volume (default true, so existing configs
    # are unaffected) skip the expensive full-mesh/full-grid branch below for
    # whichever one isn't wanted. model_type/get_num_vars above are left
    # alone -- the model itself is still loaded as combined either way, this
    # only controls which STL-resolution output gets computed and saved.
    compute_surface = cfg.eval.get("compute_surface", True)
    compute_volume = cfg.eval.get("compute_volume", True)

    if (model_type == "combined" or model_type == "surface") and compute_surface:
        surface_variable_names = list(cfg.variables.surface.solution.keys())
    else:
        surface_variable_names = []

    if (model_type == "combined" or model_type == "volume") and compute_volume:
        volume_variable_names = list(cfg.variables.volume.solution.keys())
    else:
        volume_variable_names = []

    ######################################################
    # Check that the sample size is equal.
    # unequal samples could be done but they aren't, here.s
    ######################################################
    if cfg.model.model_type == "combined":
        if cfg.model.volume_points_sample != cfg.model.surface_points_sample:
            raise ValueError(
                "Volume and surface points sample must be equal for combined model"
            )

    # Get the number of sample points:
    sample_points = (
        cfg.model.surface_points_sample
        if cfg.model.model_type == "surface"
        else cfg.model.volume_points_sample
    )

    ######################################################
    # If the batch size doesn't evenly divide
    # the num points, that's ok.  But print a warning
    # that the total points will get tweaked.
    ######################################################
    if cfg.eval.num_points % sample_points != 0:
        logger.warning(
            f"Batch size {sample_points} doesn't evenly divide num points {cfg.eval.num_points}."
        )
        logger.warning(
            f"Total points will be rounded up to {((cfg.eval.num_points // sample_points) + 1) * sample_points}."
        )

    ######################################################
    # Configure the dataset
    # We are applying preprocessing in a separate step
    # for this - so the dataset and datapipe are separate
    ######################################################

    # This helper function is to determine which keys to read from the data
    # (and which to use default values for, if they aren't present - like
    # air_density, for example)
    keys_to_read, keys_to_read_if_available = get_keys_to_read(
        cfg, model_type, get_ground_truth=True
    )
    # Override the model type
    # For the inference pipeline, we adjust the tooling a little for the data.
    # We use only a bare STL dataset that will read the mesh coordinates
    # and triangle definitions.  We'll compute the centers and normals
    # on the GPU (instead of on the CPU, as pyvista would do) and
    # then we can sample from that mesh on the GPU.
    # test_dataset = DrivaerMLDataset(
    #     data_dir=cfg.eval.test_path,
    #     keys_to_read=[
    #         "stl_coordinates",
    #         "stl_faces",
    #     ],
    #     keys_to_read_if_available=keys_to_read_if_available,
    #     output_device=dist.device,
    # )

    # Volumetric data will be generated on the fly on the GPU.

    ######################################################
    # Configure the datapipe
    # We _won't_ iterate over the datapipe, however, we can use the
    # datapipe processing tools on the sampled surface and
    # volume points with the same preprocessing.
    # It also is used to un-normalize the model outputs.
    ######################################################
    overrides = {}
    if hasattr(cfg.data, "gpu_preprocessing"):
        overrides["gpu_preprocessing"] = cfg.data.gpu_preprocessing

    if hasattr(cfg.data, "gpu_output"):
        overrides["gpu_output"] = cfg.data.gpu_output

    test_dataloader = create_domino_dataset(
        cfg,
        phase="test",
        keys_to_read=["stl_coordinates", "stl_faces"],
        keys_to_read_if_available=keys_to_read_if_available,
        vol_factors=vol_factors,
        surf_factors=surf_factors,
        normalize_coordinates=cfg.data.normalize_coordinates,
        sample_in_bbox=cfg.data.sample_in_bbox,
        sampling=cfg.data.sampling,
        device_mesh=domain_mesh,
        placements=placements,
    )

    ######################################################
    # The sampler is used in multi-gpu inference to
    # coordinate the batches used for each rank.
    ######################################################
    test_sampler = DistributedSampler(
        test_dataloader,
        num_replicas=data_replica_size,
        rank=data_rank,
        **cfg.train.sampler,
    )

    ######################################################
    # Configure the model
    # and move it to the device.
    ######################################################
    model = DoMINO(
        input_features=3,
        output_features_vol=num_vol_vars,
        output_features_surf=num_surf_vars,
        global_features=num_global_features,
        model_parameters=cfg.model,
    ).to(dist.device)

    # Print model summary (structure and parmeter count).
    logger.info(f"Model summary:\n{torchinfo.summary(model, verbose=0, depth=2)}\n")

    if dist.world_size > 1:
        torch.distributed.barrier()

    load_checkpoint(
        to_absolute_path(cfg.resume_dir),
        models=model,
        device=dist.device,
    )

    start_time = time.perf_counter()

    # This controls what indices to use for each epoch.
    test_sampler.set_epoch(0)

    prof = Profiler()

    model.eval()
    epoch_start_time = time.perf_counter()
    with prof:
        inference_epoch(
            dataloader=test_dataloader,
            sampler=test_sampler,
            model=model,
            logger=logger,
            gpu_handle=gpu_handle,
            batch_size=sample_points,
            total_points=cfg.eval.num_points,
            surface_solution_cfg=dict(cfg.variables.surface.solution)
            if surface_variable_names
            else None,
            volume_solution_cfg=dict(cfg.variables.volume.solution)
            if volume_variable_names
            else None,
            # Coarse first cut (~10 cells across the car's length) purely to
            # prove the grid/connectivity mechanism works in Kit-CAE --
            # 128*24*16 = 49152 = 6 * volume_points_sample(8192).
            volume_grid_dims=(128, 24, 16) if volume_variable_names else None,
            save_path=to_absolute_path(cfg.eval.save_path),
        )
    epoch_end_time = time.perf_counter()
    logger.info(
        f"Device {dist.device}, Epoch took {epoch_end_time - epoch_start_time:.3f} seconds"
    )


if __name__ == "__main__":
    # Profiler().enable("torch")
    # Profiler().initialize()
    main()
    # Profiler().finalize()
