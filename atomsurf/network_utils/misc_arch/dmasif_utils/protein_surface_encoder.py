import torch
import torch.nn as nn
from pykeops.torch import LazyTensor

from .geometry_processing import (
        curvatures,
)
from .helper import diagonal_ranges
from .benchmark_models import dMaSIFConv_seg


def knn_atoms(x, y, x_batch, y_batch, k):
    N, D = x.shape
    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, :, :])

    pairwise_distance_ij = ((x_i - y_j) ** 2).sum(-1)
    pairwise_distance_ij.ranges = diagonal_ranges(x_batch, y_batch)

    # N.B.: KeOps doesn't yet support backprop through Kmin reductions...
    # dists, idx = pairwise_distance_ij.Kmin_argKmin(K=k,axis=1)
    # So we have to re-compute the values ourselves:
    idx = pairwise_distance_ij.argKmin(K=k, axis=1)  # (N, K)
    x_ik = y[idx.view(-1)].view(N, k, D)
    dists = ((x[:, None, :] - x_ik) ** 2).sum(-1)

    return idx, dists


def get_atom_features(x, y, x_batch, y_batch, y_atomtype, k=16):

    idx, dists = knn_atoms(x, y, x_batch, y_batch, k=k)  # (num_points, k)
    num_points, _ = idx.size()

    idx = idx.view(-1)
    dists = 1 / dists.view(-1, 1)
    _, num_dims = y_atomtype.size()

    feature = y_atomtype[idx, :]
    feature = torch.cat([feature, dists], dim=1)
    feature = feature.view(num_points, k, num_dims + 1)

    return feature


class Atom_embedding(nn.Module):
    def __init__(self, args):
        super(Atom_embedding, self).__init__()
        self.D = args.atom_dims
        self.k = 16
        self.conv1 = nn.Linear(self.D + 1, self.D)
        self.conv2 = nn.Linear(self.D, self.D)
        self.conv3 = nn.Linear(2 * self.D, self.D)
        self.bn1 = nn.BatchNorm1d(self.D)
        self.bn2 = nn.BatchNorm1d(self.D)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atomtypes, x_batch, y_batch):
        fx = get_atom_features(x, y, x_batch, y_batch, y_atomtypes, k=self.k)
        fx = self.conv1(fx)
        fx = fx.view(-1, self.D)
        fx = self.bn1(self.relu(fx))
        fx = fx.view(-1, self.k, self.D)
        fx1 = fx.sum(dim=1, keepdim=False)

        fx = self.conv2(fx)
        fx = fx.view(-1, self.D)
        fx = self.bn2(self.relu(fx))
        fx = fx.view(-1, self.k, self.D)
        fx2 = fx.sum(dim=1, keepdim=False)
        fx = torch.cat((fx1, fx2), dim=-1)
        fx = self.conv3(fx)

        return fx


class AtomNet(nn.Module):
    def __init__(self, args):
        super(AtomNet, self).__init__()
        self.args = args

        self.transform_types = nn.Sequential(
            nn.Linear(args.atom_dims, args.atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args.atom_dims, args.atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args.atom_dims, args.atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.embed = Atom_embedding(args)

    def forward(self, xyz, atom_xyz, atomtypes, batch, atom_batch):
        # Run a DGCNN on the available information:
        atomtypes = self.transform_types(atomtypes)
        return self.embed(xyz, atom_xyz, atomtypes, batch, atom_batch)

class Atom_embedding_MP(nn.Module):
    def __init__(self, args):
        super(Atom_embedding_MP, self).__init__()
        self.D = args.atom_dims
        self.k = 16
        self.n_layers = 3
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.D + 1, 2 * self.D + 1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(2 * self.D + 1, self.D),
                )
                for i in range(self.n_layers)
            ]
        )
        self.norm = nn.ModuleList(
            [nn.GroupNorm(2, self.D) for i in range(self.n_layers)]
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atomtypes, x_batch, y_batch):
        idx, dists = knn_atoms(x, y, x_batch, y_batch, k=self.k)  # N, 9, 7
        num_points = x.shape[0]
        num_dims = y_atomtypes.shape[-1]

        point_emb = torch.ones_like(x[:, 0])[:, None].repeat(1, num_dims)
        for i in range(self.n_layers):
            features = y_atomtypes[idx.reshape(-1), :]
            features = torch.cat([features, dists.reshape(-1, 1)], dim=1)
            features = features.view(num_points, self.k, num_dims + 1)
            features = torch.cat(
                [point_emb[:, None, :].repeat(1, self.k, 1), features], dim=-1
            )  # N, 8, 13

            messages = self.mlp[i](features)  # N,8,6
            messages = messages.sum(1)  # N,6
            point_emb = point_emb + self.relu(self.norm[i](messages))

        return point_emb

class Atom_Atom_embedding_MP(nn.Module):
    def __init__(self, args):
        super(Atom_Atom_embedding_MP, self).__init__()
        self.D = args.atom_dims
        self.k = 17
        self.n_layers = 3

        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.D + 1, 2 * self.D + 1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(2 * self.D + 1, self.D),
                )
                for i in range(self.n_layers)
            ]
        )

        self.norm = nn.ModuleList(
            [nn.GroupNorm(2, self.D) for i in range(self.n_layers)]
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atomtypes, x_batch, y_batch):
        idx, dists = knn_atoms(x, y, x_batch, y_batch, k=self.k)  # N, 9, 7
        idx = idx[:, 1:]  # Remove self
        dists = dists[:, 1:]
        k = self.k - 1
        num_points = y_atomtypes.shape[0]

        out = y_atomtypes
        for i in range(self.n_layers):
            _, num_dims = out.size()
            features = out[idx.reshape(-1), :]
            features = torch.cat([features, dists.reshape(-1, 1)], dim=1)
            features = features.view(num_points, k, num_dims + 1)
            features = torch.cat(
                [out[:, None, :].repeat(1, k, 1), features], dim=-1
            )  # N, 8, 13

            messages = self.mlp[i](features)  # N,8,6
            messages = messages.sum(1)  # N,6
            out = out + self.relu(self.norm[i](messages))

        return out

class AtomNet_MP(nn.Module):
    def __init__(self, args):
        super(AtomNet_MP, self).__init__()
        self.args = args

        self.transform_types = nn.Sequential(
            nn.Linear(args.atom_dims, args.atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args.atom_dims, args.atom_dims),
        )

        self.embed = Atom_embedding_MP(args)
        self.atom_atom = Atom_Atom_embedding_MP(args)

    def forward(self, xyz, atom_xyz, atomtypes, batch, atom_batch):
        # Run a DGCNN on the available information:
        atomtypes = self.transform_types(atomtypes)
        atomtypes = self.atom_atom(
            atom_xyz, atom_xyz, atomtypes, atom_batch, atom_batch
        )
        atomtypes = self.embed(xyz, atom_xyz, atomtypes, batch, atom_batch)
        return atomtypes

class dMaSIF(nn.Module):
    def __init__(self, args):
        super(dMaSIF, self).__init__()
        # Additional geometric features: mean and Gauss curvatures computed at different scales.
        self.curvature_scales = args.curvature_scales
        self.args = args

        I = args.in_channels
        O = args.orientation_units
        E = args.emb_dims

        # Computes chemical features
        self.atomnet = AtomNet_MP(args)
        self.dropout = nn.Dropout(args.dropout)

        # Post-processing, without batch norm:
        self.orientation_scores = nn.Sequential(
            nn.Linear(I, O),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(O, 1),
        )

        # Segmentation network:
        self.conv = dMaSIFConv_seg(
            args,
            in_channels=I,
            out_channels=E,
            n_layers=args.n_layers,
            radius=args.radius,
        )


    def features(self, P):
        """Estimates geometric and chemical features from a protein surface or a cloud of atoms."""

        # Estimate the curvatures using the triangles or the estimated normals:
        P_curvatures = curvatures(
            P["xyz"],
            triangles= None,
            normals= P["normals"],
            scales=self.curvature_scales,
            batch=P["batch"],
        )
        # Compute chemical features on-the-fly:
        chemfeats = self.atomnet(
            P["xyz"], P["atom_xyz"], P["atomtypes"], P["batch"], P["batch_atoms"]
        )
        
        return torch.cat([P_curvatures, chemfeats], dim=1).contiguous()


    def forward(self, P):
        """Embeds all points of a protein in a high-dimensional vector space."""
       
        features = self.dropout(self.features(P))
        P["input_features"] = features

        torch.cuda.synchronize(device=features.device)

        self.conv.load_mesh(
            P["xyz"],
            triangles= None,
            normals= P["normals"],
            weights=self.orientation_scores(features),
            batch=P["batch"],
        )
        P["embedding"] = self.conv(features)

        return P
