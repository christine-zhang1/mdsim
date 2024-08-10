"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch_geometric.nn import SchNet, radius_graph
from torch_scatter import scatter
from torch_sparse import SparseTensor

from mdsim.common.registry import registry
from mdsim.common.utils import (
    compute_neighbors,
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)

from mdsim.models.gemnet.utils import (
    inner_product_normalized,
    mask_neighbors,
    ragged_range,
    repeat_blocks,
)

from mdsim.models.gemnet.layers.atom_update_block import OutputBlock
from mdsim.models.gemnet.layers.efficient import EfficientInteractionDownProjection
from mdsim.models.gemnet.layers.interaction_block import InteractionBlockTripletsOnly
from mdsim.models.gemnet.layers.base_layers import Dense
from mdsim.models.gemnet.layers.radial_basis import RadialBasis
from mdsim.models.gemnet.layers.spherical_basis import CircularBasisLayer
from mdsim.models.gemnet.layers.embedding_block import AtomEmbedding, EdgeEmbedding

@registry.register_model("schnet")
class SchNetWrap(SchNet):
    r"""Wrapper around the continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_. Each layer uses interaction
    block of the form:

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    Args:
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets (int): Number of targets to predict.
        use_pbc (bool, optional): If set to :obj:`True`, account for periodic boundary conditions.
            (default: :obj:`True`)
        regress_forces (bool, optional): If set to :obj:`True`, predict forces by differentiating
            energy with respect to positions.
            (default: :obj:`True`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
        hidden_channels (int, optional): Number of hidden channels.
            (default: :obj:`128`)
        num_filters (int, optional): Number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): Number of interaction blocks
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
    """

    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,
        use_pbc=True,
        regress_forces=True,
        otf_graph=False,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0, # becomes 5.0 from schnet.yml file, which is the same as in gemnet-dT.yml
        readout="add",
        direct_forces=False,
    ):
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.direct_forces = direct_forces # when doing direct forces, keep regress_forces as true so it calculates force losses. see gemnet

        # copied these from gemnet-dT.yml file for aspirin
        num_radial = 6
        rbf = {"name": "gaussian"}
        envelope = {"name": "polynomial", "exponent": 5}
        num_spherical = 7
        cbf = {"name": "spherical_harmonics"}
        emb_size_atom = 128
        emb_size_edge = 128
        activation = "silu"
        emb_size_rbf = 16
        emb_size_cbf = 16
        emb_size_trip = 64
        emb_size_bil_trip = 64
        num_before_skip = 1
        num_after_skip = 1
        num_concat = 1
        num_atom = 2
        scale_file = "configs/md17/gemnet-dT-scale.json"
        output_init = "HeOrthogonal"

        super(SchNetWrap, self).__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout,
        )

        # initializing with the values from gemnet
        self.radial_basis = RadialBasis(
            num_radial=num_radial,
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        radial_basis_cbf3 = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        )
        self.cbf_basis3 = CircularBasisLayer(
            num_spherical,
            radial_basis=radial_basis_cbf3,
            cbf=cbf,
            efficient=True,
        )

        ### ------------------------------- Share Down Projections ------------------------------ ###
        # Share down projection across all interaction blocks
        self.mlp_rbf3 = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_cbf3 = EfficientInteractionDownProjection(
            num_spherical, num_radial, emb_size_cbf
        )

        # Share the dense Layer of the atom embedding block accross the interaction blocks
        self.mlp_rbf_h = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_rbf_out = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        ### ------------------------------------------------------------------------------------- ###

        # Embedding block
        self.atom_emb = AtomEmbedding(emb_size_atom)
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation
        )

        # Interaction blocks
        int_blocks = []
        for i in range(2):
            int_blocks.append(InteractionBlockTripletsOnly(
                emb_size_atom=emb_size_atom,
                emb_size_edge=emb_size_edge,
                emb_size_trip=emb_size_trip,
                emb_size_rbf=emb_size_rbf,
                emb_size_cbf=emb_size_cbf,
                emb_size_bil_trip=emb_size_bil_trip,
                num_before_skip=num_before_skip,
                num_after_skip=num_after_skip,
                num_concat=num_concat,
                num_atom=num_atom,
                activation=activation,
                scale_file=scale_file,
                name=f"Gemnet_IntBlock_{i+1}",
            ))

        out_blocks = []
        for i in range(3):
            out_blocks.append(OutputBlock(emb_size_atom=emb_size_atom,
                emb_size_edge=emb_size_edge,
                emb_size_rbf=emb_size_rbf,
                nHidden=num_atom,
                num_targets=num_targets,
                activation=activation,
                output_init=output_init,
                direct_forces=direct_forces,
                scale_file=scale_file,
                name=f"OutBlock_{i}",
            ))

        self.gemnet_int_blocks = torch.nn.ModuleList(int_blocks)
        self.gemnet_out_blocks = torch.nn.ModuleList(out_blocks)

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch

        forces = None
        if self.otf_graph:
            edge_index, cell_offsets, _, neighbors = radius_graph_pbc(
                data, self.cutoff, 500
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        if self.use_pbc: # false, so this if chunk is unused
            assert z.dim() == 1 and z.dtype == torch.long

            out = get_pbc_distances(
                data.pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.natoms,
            )

            edge_index = out["edge_index"]
            edge_weight = out["distances"]
            edge_attr = self.distance_expansion(edge_weight)

            h = self.embedding(z)
            for interaction in self.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)

            h = self.lin1(h)
            h = self.act(h)
            h = self.lin2(h)

            batch = torch.zeros_like(z) if batch is None else batch
            energy = scatter(h, batch, dim=0, reduce=self.readout)
        else:
            # copying code over from gemnet
            (
                edge_index,
                neighbors,
                D_st,
                V_st,
                id_swap,
                id3_ba,
                id3_ca,
                id3_ragged_idx,
            ) = self.generate_interaction_graph_simple(data)
            idx_s, idx_t = edge_index

            # Calculate triplet angles.
            cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
            rad_cbf3, cbf3 = self.cbf_basis3(D_st, cosφ_cab, id3_ca)

            rbf = self.radial_basis(D_st)

            # Embedding block
            h = self.atom_emb(data.atomic_numbers.long())
            # (nAtoms, emb_size_atom)
            m = self.edge_emb(h, rbf, idx_s, idx_t)  # (nEdges, emb_size_edge)

            rbf3 = self.mlp_rbf3(rbf)
            cbf3 = self.mlp_cbf3(rad_cbf3, cbf3, id3_ca, id3_ragged_idx)

            rbf_h = self.mlp_rbf_h(rbf)
            rbf_out = self.mlp_rbf_out(rbf)

            E_t, F_st = self.gemnet_out_blocks[0](h, m, rbf_out, idx_t)

            for i in range(2):
                h, m = self.gemnet_int_blocks[i](
                    h=h,
                    m=m,
                    rbf3=rbf3,
                    cbf3=cbf3,
                    id3_ragged_idx=id3_ragged_idx,
                    id_swap=id_swap,
                    id3_ba=id3_ba,
                    id3_ca=id3_ca,
                    rbf_h=rbf_h,
                    idx_s=idx_s,
                    idx_t=idx_t,
                )

                E, F = self.gemnet_out_blocks[i + 1](h, m, rbf_out, idx_t)
                # (nAtoms, num_targets), (nEdges, num_targets)
                F_st += F
                E_t += E

            energy, forces = super(SchNetWrap, self).forward(z, pos, batch, direct_forces=self.direct_forces, h=h, m=m, F_st=F_st, E_t=E_t)
        return energy, forces

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy, forces = self._forward(data)

        if self.regress_forces:
            if not self.direct_forces:
                forces = -1 * (
                    torch.autograd.grad(
                        energy,
                        data.pos,
                        grad_outputs=torch.ones_like(energy),
                        create_graph=True,
                    )[0]
                )
            return energy, forces
        else:
            return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    ### ------------------------ GEMNET FUNCTIONS ------------------------ ###

    def generate_interaction_graph_simple(self, data):
        num_atoms = data.atomic_numbers.size(0)

        # cut out the first if chunk that checks if self.use_pbc is true
        edge_index = radius_graph(
            data.pos,
            r=self.cutoff,
            batch=data.batch,
            max_num_neighbors=32,
        )
        j, i = edge_index
        distance_vec = data.pos[j] - data.pos[i]

        D_st = distance_vec.norm(dim=-1)
        V_st = -distance_vec / D_st[:, None]
        cell_offsets = torch.zeros(
            edge_index.shape[1], 3, device=data.pos.device
        )
        neighbors = compute_neighbors(data, edge_index)

        # Mask interaction edges if required
        if self.otf_graph or np.isclose(self.cutoff, 6):
            select_cutoff = None
        else:
            select_cutoff = self.cutoff
        (edge_index, cell_offsets, neighbors, D_st, V_st,) = self.select_edges(
            data=data,
            edge_index=edge_index,
            cell_offsets=cell_offsets,
            neighbors=neighbors,
            edge_dist=D_st,
            edge_vector=V_st,
            cutoff=select_cutoff,
        )
        
        (
            edge_index,
            cell_offsets,
            neighbors,
            D_st,
            V_st,
        ) = self.reorder_symmetric_edges(
            edge_index, cell_offsets, neighbors, D_st, V_st
        )

        # Indices for swapping c->a and a->c (for symmetric MP)
        block_sizes = neighbors // 2
        id_swap = repeat_blocks(
            block_sizes,
            repeats=2,
            continuous_indexing=False,
            start_idx=block_sizes[0],
            block_inc=block_sizes[:-1] + block_sizes[1:],
            repeat_inc=-block_sizes,
        )

        id3_ba, id3_ca, id3_ragged_idx = self.get_triplets(
            edge_index, num_atoms=num_atoms
        )

        return (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        )

    def select_edges(
        self,
        data,
        edge_index,
        cell_offsets,
        neighbors,
        edge_dist,
        edge_vector,
        cutoff=None,
    ):
        if cutoff is not None:
            edge_mask = edge_dist <= cutoff

            edge_index = edge_index[:, edge_mask]
            cell_offsets = cell_offsets[edge_mask]
            neighbors = mask_neighbors(neighbors, edge_mask)
            edge_dist = edge_dist[edge_mask]
            edge_vector = edge_vector[edge_mask]

        empty_image = neighbors == 0
        if torch.any(empty_image):
            raise ValueError(
                f"An image has no neighbors: id={data.id[empty_image]}, "
                f"sid={data.sid[empty_image]}, fid={data.fid[empty_image]}"
            )
        return edge_index, cell_offsets, neighbors, edge_dist, edge_vector

    def reorder_symmetric_edges(
        self, edge_index, cell_offsets, neighbors, edge_dist, edge_vector
    ):
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running torch.unique.
        But this does not seem worth it.
        """

        # Generate mask
        mask_sep_atoms = edge_index[0] < edge_index[1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | (
                (cell_offsets[:, 0] == 0)
                & (cell_offsets[:, 1] == 0)
                & (cell_offsets[:, 2] < 0)
            )
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [
                edge_index_new,
                torch.stack([edge_index_new[1], edge_index_new[0]], dim=0),
            ],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device),
            neighbors,
        )
        
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * torch.bincount(
            batch_edge, minlength=neighbors.size(0)
        )

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(
            cell_offsets, mask, edge_reorder_idx, True
        )
        edge_dist_new = self.select_symmetric_edges(
            edge_dist, mask, edge_reorder_idx, False
        )
        edge_vector_new = self.select_symmetric_edges(
            edge_vector, mask, edge_reorder_idx, True
        )

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_dist_new,
            edge_vector_new,
        )

    def get_triplets(self, edge_index, num_atoms):
        """
        Get all b->a for each edge c->a.
        It is possible that b=c, as long as the edges are distinct.

        Returns
        -------
        id3_ba: torch.Tensor, shape (num_triplets,)
            Indices of input edge b->a of each triplet b->a<-c
        id3_ca: torch.Tensor, shape (num_triplets,)
            Indices of output edge c->a of each triplet b->a<-c
        id3_ragged_idx: torch.Tensor, shape (num_triplets,)
            Indices enumerating the copies of id3_ca for creating a padded matrix
        """
        idx_s, idx_t = edge_index  # c->a (source=c, target=a)

        value = torch.arange(
            idx_s.size(0), device=idx_s.device, dtype=idx_s.dtype
        )
        # Possibly contains multiple copies of the same edge (for periodic interactions)
        adj = SparseTensor(
            row=idx_t,
            col=idx_s,
            value=value,
            sparse_sizes=(num_atoms, num_atoms),
        )
        adj_edges = adj[idx_t]

        # Edge indices (b->a, c->a) for triplets.
        id3_ba = adj_edges.storage.value()
        id3_ca = adj_edges.storage.row()

        # Remove self-loop triplets
        # Compare edge indices, not atom indices to correctly handle periodic interactions
        mask = id3_ba != id3_ca
        id3_ba = id3_ba[mask]
        id3_ca = id3_ca[mask]

        # Get indices to reshape the neighbor indices b->a into a dense matrix.
        # id3_ca has to be sorted for this to work.
        num_triplets = torch.bincount(id3_ca, minlength=idx_s.size(0))
        id3_ragged_idx = ragged_range(num_triplets)

        return id3_ba, id3_ca, id3_ragged_idx

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered