"""
GraphBuilder — convert tabular player/entity data into torch_geometric graphs

Priority 2 from ResearchForge build plan.

Each row in the dataframe represents one entity (player/ball) in a game frame.
Each game frame becomes one torch_geometric.data.Data object.

Node features : [x, y, vx, vy, is_ball, team_id]
Edges         : between entities within proximity_threshold distance
Edge features : [distance, relative_velocity_magnitude]
Label         : binary/multiclass from label_col (same for all nodes in a frame)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional


class GraphBuilder:
    """
    Convert a flat player-tracking DataFrame into a list of
    torch_geometric.data.Data objects — one per game frame.

    Usage::

        builder = GraphBuilder()
        graphs = builder.build_from_dataframe(
            df,
            position_cols=["x", "y", "velocity_x", "velocity_y"],
            label_col="shot_made",
            frame_col="frame_id",          # optional grouping column
            team_col="team_id",            # optional
            is_ball_col="is_ball",         # optional, bool column
            proximity_threshold=3.0,
        )
    """

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        position_cols: List[str],
        label_col: str,
        frame_col: Optional[str] = None,
        team_col: Optional[str] = None,
        is_ball_col: Optional[str] = None,
        proximity_threshold: float = 3.0,
    ) -> list:
        """
        Parameters
        ----------
        df               : DataFrame with one row per entity per frame
        position_cols    : column names for [x, y, vx, vy] in that order
                           (at minimum [x, y]; vx/vy optional)
        label_col        : target column (same value replicated per frame)
        frame_col        : column identifying each game frame; if None,
                           the entire df is treated as one frame
        team_col         : integer column with team id (0 or 1)
        is_ball_col      : boolean / 0-1 column marking the ball entity
        proximity_threshold : max Euclidean distance (metres) to draw an edge
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError as e:
            raise ImportError(
                "PyTorch Geometric is required for graph building. "
                "Install with: pip install torch_geometric\n"
                f"Original error: {e}"
            ) from e

        if frame_col and frame_col in df.columns:
            frames = df.groupby(frame_col)
        else:
            frames = [(0, df)]

        graph_list: list[Data] = []

        for frame_id, frame_df in frames:
            frame_df = frame_df.reset_index(drop=True)
            n_nodes = len(frame_df)
            if n_nodes == 0:
                continue

            # ── Node features ──────────────────────────────────────
            feature_parts = []

            # x, y (required)
            if len(position_cols) >= 2:
                xy = frame_df[position_cols[:2]].values.astype(np.float32)
                feature_parts.append(xy)
            else:
                raise ValueError(
                    f"position_cols must have at least 2 entries (x, y). Got: {position_cols}"
                )

            # vx, vy (optional)
            if len(position_cols) >= 4:
                vxy = frame_df[position_cols[2:4]].values.astype(np.float32)
            else:
                vxy = np.zeros((n_nodes, 2), dtype=np.float32)
            feature_parts.append(vxy)

            # is_ball flag
            if is_ball_col and is_ball_col in frame_df.columns:
                is_ball = frame_df[is_ball_col].values.astype(np.float32).reshape(-1, 1)
            else:
                # Heuristic: first row = ball by default
                is_ball = np.zeros((n_nodes, 1), dtype=np.float32)
                is_ball[0, 0] = 1.0
            feature_parts.append(is_ball)

            # team_id
            if team_col and team_col in frame_df.columns:
                team = frame_df[team_col].values.astype(np.float32).reshape(-1, 1)
            else:
                team = np.zeros((n_nodes, 1), dtype=np.float32)
            feature_parts.append(team)

            node_features = np.concatenate(feature_parts, axis=1)  # (N, 6)
            x = torch.from_numpy(node_features)

            # ── Edges: proximity-based ──────────────────────────────
            positions = xy  # shape (N, 2)
            edge_src, edge_dst, edge_attrs = [], [], []

            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i == j:
                        continue
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist = float(np.sqrt(dx * dx + dy * dy))
                    if dist <= proximity_threshold:
                        edge_src.append(i)
                        edge_dst.append(j)
                        # Edge features: [distance, rel_velocity_magnitude]
                        dvx = vxy[i, 0] - vxy[j, 0]
                        dvy = vxy[i, 1] - vxy[j, 1]
                        rel_vel = float(np.sqrt(dvx * dvx + dvy * dvy))
                        edge_attrs.append([dist, rel_vel])

            if edge_src:
                edge_index = torch.tensor(
                    [edge_src, edge_dst], dtype=torch.long
                )
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
            else:
                # No edges — fully isolated nodes (still valid for some GNNs)
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr  = torch.zeros((0, 2),  dtype=torch.float32)

            # ── Label ───────────────────────────────────────────────
            label_values = frame_df[label_col].values
            # Use the most common label in the frame as the frame-level label
            label_value  = np.bincount(
                label_values.astype(int) if label_values.dtype != object
                else np.array([int(v) for v in label_values])
            ).argmax()
            y = torch.tensor([label_value], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.frame_id = frame_id
            graph_list.append(data)

        return graph_list

    def summary(self, graph_list: list) -> str:
        """Print a summary of the constructed graph dataset."""
        if not graph_list:
            return "Empty graph list"
        total_nodes = sum(g.x.shape[0] for g in graph_list)
        total_edges = sum(g.edge_index.shape[1] for g in graph_list)
        node_dim    = graph_list[0].x.shape[1]
        edge_dim    = graph_list[0].edge_attr.shape[1] if graph_list[0].edge_attr is not None else 0
        return (
            f"GraphDataset: {len(graph_list)} graphs | "
            f"{total_nodes} total nodes ({total_nodes/len(graph_list):.1f} avg/graph) | "
            f"{total_edges} total edges | "
            f"node_dim={node_dim} edge_dim={edge_dim}"
        )
