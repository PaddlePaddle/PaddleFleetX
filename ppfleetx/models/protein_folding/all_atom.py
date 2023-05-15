#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Dict, Optional
import paddle

from .common import (batched_gather, )

from . import (
    residue_constants,
    r3, )


def get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
        A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
        in the order specified in residue_constants.restypes + unknown residue type
        at the end. For chi angles which are not defined on the residue, the
        positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in residue_constants.restypes:
        residue_name = residue_constants.restype_1to3[residue_name]
        residue_chi_angles = residue_constants.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append(
                [residue_constants.atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append(
                [0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return paddle.to_tensor(chi_atom_indices)


def atom37_to_torsion_angles(
        aatype: paddle.Tensor,  # (B, T, N)
        all_atom_pos: paddle.Tensor,  # (B, T, N, 37, 3)
        all_atom_mask: paddle.Tensor,  # (B, T, N, 37)
        placeholder_for_undefined=False, ) -> Dict[str, paddle.Tensor]:
    """Computes the 7 torsion angles (in sin, cos encoding) for each residue.

    The 7 torsion angles are in the order
    '[pre_omega, phi, psi, chi_1, chi_2, chi_3, chi_4]',
    here pre_omega denotes the omega torsion angle between the given amino acid
    and the previous amino acid.

    Args:
        aatype: Amino acid type, given as array with integers.
        all_atom_pos: atom37 representation of all atom coordinates.
        all_atom_mask: atom37 representation of mask on all atom coordinates.
        placeholder_for_undefined: flag denoting whether to set masked torsion
        angles to zero.
    Returns:
        Dict containing:
        * 'torsion_angles_sin_cos': Array with shape (B, N, 7, 2) where the final
            2 dimensions denote sin and cos respectively
        * 'alt_torsion_angles_sin_cos': same as 'torsion_angles_sin_cos', but
            with the angle shifted by pi for all chi angles affected by the naming
            ambiguities.
        * 'torsion_angles_mask': Mask for which chi angles are present.
    """

    # Map aatype > 20 to 'Unknown' (20).
    aatype = paddle.minimum(
        aatype.astype('int'),
        paddle.full(
            shape=[1], fill_value=20, dtype='int'))

    num_batch, num_temp, num_res = aatype.shape

    # Compute the backbone angles.
    pad = paddle.zeros([num_batch, num_temp, 1, 37, 3])
    prev_all_atom_pos = paddle.concat(
        [pad, all_atom_pos[..., :-1, :, :]], axis=-3)

    pad = paddle.zeros([num_batch, num_temp, 1, 37])
    prev_all_atom_mask = paddle.concat(
        [pad, all_atom_mask[..., :-1, :]], axis=-2)

    # For each torsion angle collect the 4 atom positions that define this angle.
    # shape (B, T, N, atoms=4, xyz=3)
    pre_omega_atom_pos = paddle.concat(
        [
            prev_all_atom_pos[..., 1:3, :],  # prev CA, C
            all_atom_pos[..., 0:2, :]  # this N, CA
        ],
        axis=-2)

    phi_atom_pos = paddle.concat(
        [
            prev_all_atom_pos[..., 2:3, :],  # prev C
            all_atom_pos[..., 0:3, :]  # this N, CA, C
        ],
        axis=-2)

    psi_atom_pos = paddle.concat(
        [
            all_atom_pos[..., 0:3, :],  # this N, CA, C
            all_atom_pos[..., 4:5, :]  # this O
        ],
        axis=-2)

    # Collect the masks from these atoms.
    # Shape [batch, n_temp, num_res]
    pre_omega_mask = (
        paddle.prod(
            prev_all_atom_mask[..., 1:3], axis=-1)  # prev CA, C
        * paddle.prod(
            all_atom_mask[..., 0:2], axis=-1))  # this N, CA
    phi_mask = (
        prev_all_atom_mask[..., 2]  # prev C
        * paddle.prod(
            all_atom_mask[..., 0:3], axis=-1))  # this N, CA, C
    psi_mask = (
        paddle.prod(
            all_atom_mask[..., 0:3], axis=-1) *  # this N, CA, C
        all_atom_mask[..., 4])  # this O

    # Collect the atoms for the chi-angles.
    # Compute the table of chi angle indices. Shape: [restypes, chis=4, atoms=4].
    chi_atom_indices = get_chi_atom_indices()

    # Select atoms to compute chis. Shape: [batch, num_temp, num_res, chis=4, atoms=4].
    atom_indices = batched_gather(
        params=chi_atom_indices, indices=aatype, axis=0, batch_dims=0)

    # Gather atom positions. Shape: [batch, num_temp, num_res, chis=4, atoms=4, xyz=3].
    chis_atom_pos = batched_gather(
        params=all_atom_pos, indices=atom_indices, axis=0, batch_dims=3)

    # Copy the chi angle mask, add the UNKNOWN residue. Shape: [restypes, 4].
    chi_angles_mask = list(residue_constants.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = paddle.to_tensor(chi_angles_mask)

    # Compute the chi angle mask. I.e. which chis angles exist according to the
    # aatype. Shape [batch, num_temp, num_res, chis=4].
    chis_mask = batched_gather(
        params=chi_angles_mask, indices=aatype, axis=0, batch_dims=0)
    # Constrain the chis_mask to those chis, where the ground truth coordinates of
    # all defining four atoms are available.
    # Gather the chi angle atoms mask. Shape: [batch, num_temp, num_res, chis=4, atoms=4].
    chi_angle_atoms_mask = batched_gather(
        params=all_atom_mask, indices=atom_indices, axis=0, batch_dims=3)
    # Check if all 4 chi angle atoms were set. Shape: [batch, num_temp, num_res, chis=4].
    chi_angle_atoms_mask = paddle.prod(chi_angle_atoms_mask, axis=[-1])
    chis_mask = chis_mask * chi_angle_atoms_mask

    # Stack all torsion angle atom positions.
    # Shape (B, T, N, torsions=7, atoms=4, xyz=3)
    torsions_atom_pos = paddle.concat(
        [
            pre_omega_atom_pos.unsqueeze(axis=-3),  # [:, :, :, None, :, :]
            phi_atom_pos.unsqueeze(axis=-3),  # [:, :, :, None, :, :]
            psi_atom_pos.unsqueeze(axis=-3),  # [:, :, :, None, :, :]
            chis_atom_pos
        ],
        axis=3)

    # Stack up masks for all torsion angles.
    # shape (B, T, N, torsions=7)
    torsion_angles_mask = paddle.concat(
        [
            pre_omega_mask.unsqueeze(axis=-1),  # [..., None]
            phi_mask.unsqueeze(axis=-1),  # [..., None]
            psi_mask.unsqueeze(axis=-1),  # [..., None]
            chis_mask
        ],
        axis=-1)

    # Create a frame from the first three atoms:
    # First atom: point on x-y-plane
    # Second atom: point on negative x-axis
    # Third atom: origin
    # r3.Rigids (B, T, N, torsions=7)
    torsion_frames = r3.rigids_from_3_points_vecs(
        point_on_neg_x_axis=r3.Vecs(torsions_atom_pos[..., 1, :]),
        origin=r3.Vecs(torsions_atom_pos[..., 2, :]),
        point_on_xy_plane=r3.Vecs(torsions_atom_pos[..., 0, :]))

    # Compute the position of the forth atom in this frame (y and z coordinate
    # define the chi angle)
    # r3.Vecs (B, T, N, torsions=7)
    forth_atom_rel_pos = r3.rigids_mul_vecs(
        r3.invert_rigids(torsion_frames),
        r3.vecs_from_tensor(torsions_atom_pos[..., 3, :]))

    # Normalize to have the sin and cos of the torsion angle.
    # paddle.Tensor (B, T, N, torsions=7, sincos=2)
    torsion_angles_sin_cos = paddle.stack(
        [forth_atom_rel_pos.z, forth_atom_rel_pos.y], axis=-1)
    torsion_angles_sin_cos /= paddle.sqrt(
        paddle.sum(paddle.square(torsion_angles_sin_cos),
                   axis=-1,
                   keepdim=True) + 1e-8)

    # Mirror psi, because we computed it from the Oxygen-atom.
    torsion_angles_sin_cos *= paddle.to_tensor(
        [1., 1., -1., 1., 1., 1., 1.]).reshape(
            [1, 1, 1, 7, 1])  # [None, None, None, :, None]

    # Create alternative angles for ambiguous atom names.
    chi_is_ambiguous = batched_gather(
        paddle.to_tensor(residue_constants.chi_pi_periodic), aatype)
    # chi_is_ambiguous (B, T, N, torsions=4)
    mirror_torsion_angles = paddle.concat(
        [
            paddle.ones([num_batch, num_temp, num_res, 3]),
            1.0 - 2.0 * chi_is_ambiguous
        ],
        axis=-1)
    # mirror_torsion_angles (B, T, N, torsions=7)
    alt_torsion_angles_sin_cos = torsion_angles_sin_cos * mirror_torsion_angles.unsqueeze(
        axis=-1)  # [:, :, :, :, None]

    if placeholder_for_undefined:
        # Add placeholder torsions in place of undefined torsion angles
        # (e.g. N-terminus pre-omega)
        placeholder_torsions = paddle.stack(
            [
                paddle.ones(torsion_angles_sin_cos.shape[:-1]),
                paddle.zeros(torsion_angles_sin_cos.shape[:-1])
            ],
            axis=-1)
        torsion_angles_sin_cos = torsion_angles_sin_cos * torsion_angles_mask.unsqueeze(
            axis=-1) + placeholder_torsions * (
                1 - torsion_angles_mask.unsqueeze(axis=-1))
        alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos * torsion_angles_mask.unsqueeze(
            axis=-1) + placeholder_torsions * (
                1 - torsion_angles_mask.unsqueeze(axis=-1))

    return {
        'torsion_angles_sin_cos': torsion_angles_sin_cos,  # (B, T, N, 7, 2)
        'alt_torsion_angles_sin_cos':
        alt_torsion_angles_sin_cos,  # (B, T, N, 7, 2)
        'torsion_angles_mask': torsion_angles_mask  # (B, T, N, 7)
    }
