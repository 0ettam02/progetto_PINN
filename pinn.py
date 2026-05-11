import torch
import torch.nn as nn


def _group_norm_groups(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        if channels < 1:
            raise ValueError("channels deve essere positivo.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout deve essere in [0, 1).")

        groups = _group_norm_groups(channels)
        layers: list[nn.Module] = [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="circular"),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        layers.extend(
            [
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="circular"),
                nn.GroupNorm(groups, channels),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class AdvectionCNN(nn.Module):
    """Residual CNN per forecast full-field da cloud rasterizzato su griglia regolare."""

    def __init__(
        self,
        lookback: int,
        rollout: int,
        grid_points: torch.Tensor,
        grid_shape: tuple[int, int],
        hidden_dim: int = 64,
        num_layers: int = 6,
        dropout: float = 0.0,
        interpolation_neighbors: int = 8,
        use_coordinates: bool = True,
    ):
        super().__init__()
        if lookback < 1:
            raise ValueError("lookback deve essere almeno 1.")
        if rollout < 1:
            raise ValueError("rollout deve essere almeno 1.")
        if num_layers < 1:
            raise ValueError("num_layers deve essere almeno 1.")
        if interpolation_neighbors < 1:
            raise ValueError("interpolation_neighbors deve essere positivo.")

        grid_points = torch.as_tensor(grid_points, dtype=torch.float32)
        expected_points = int(grid_shape[0] * grid_shape[1])
        if grid_points.shape != (expected_points, 2):
            raise ValueError(
                "grid_points deve avere shape "
                f"({expected_points}, 2), ricevuta {tuple(grid_points.shape)}."
            )

        self.lookback = lookback
        self.rollout = rollout
        self.grid_shape = grid_shape
        self.interpolation_neighbors = interpolation_neighbors
        self.use_coordinates = use_coordinates
        self.register_buffer("grid_points", grid_points)

        in_channels = lookback + (2 if use_coordinates else 0)
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, padding_mode="circular"),
            nn.GroupNorm(_group_norm_groups(hidden_dim), hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualConvBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, rollout, kernel_size=1),
        )

    def forward(self, grid_state: torch.Tensor) -> torch.Tensor:
        x = self._append_coordinates(grid_state)
        features = self.blocks(self.lift(x))
        delta = self.head(features)
        persistence = grid_state[:, -1:, :, :].expand(-1, self.rollout, -1, -1)
        return persistence + delta

    def forward_cloud(self, u0: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        grid_state = self.cloud_to_grid(u0.transpose(1, 2), pos)
        pred_grid = self.forward(grid_state)
        return self.grid_to_cloud(pred_grid, pos)

    def cloud_to_grid(self, values: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        interpolated = self._idw_interpolate(
            values=values,
            source_points=pos,
            target_points=self.grid_points,
        )
        batch_size, channels, _ = interpolated.shape
        height, width = self.grid_shape
        return interpolated.reshape(batch_size, channels, height, width)

    def grid_to_cloud(self, grid_values: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = grid_values.shape
        if (height, width) != self.grid_shape:
            raise ValueError(
                f"grid_values ha shape spaziale {(height, width)}, attesa {self.grid_shape}."
            )
        flat_values = grid_values.reshape(batch_size, channels, height * width)
        return self._idw_interpolate(
            values=flat_values,
            source_points=self.grid_points,
            target_points=pos,
        )

    def _append_coordinates(self, grid_state: torch.Tensor) -> torch.Tensor:
        if not self.use_coordinates:
            return grid_state
        height, width = self.grid_shape
        coords = self.grid_points.T.reshape(1, 2, height, width)
        coords = coords.to(device=grid_state.device, dtype=grid_state.dtype)
        coords = coords.expand(grid_state.shape[0], -1, -1, -1)
        return torch.cat([grid_state, coords], dim=1)

    def _idw_interpolate(
        self,
        values: torch.Tensor,
        source_points: torch.Tensor,
        target_points: torch.Tensor,
    ) -> torch.Tensor:
        if source_points.ndim == 2:
            source_points = source_points.unsqueeze(0).expand(values.shape[0], -1, -1)
        if target_points.ndim == 2:
            target_points = target_points.unsqueeze(0).expand(values.shape[0], -1, -1)

        source_points = source_points.to(device=values.device, dtype=values.dtype)
        target_points = target_points.to(device=values.device, dtype=values.dtype)
        neighbors = min(self.interpolation_neighbors, source_points.shape[1])

        distances = torch.cdist(target_points, source_points)
        nearest_dist, nearest_idx = torch.topk(
            distances,
            k=neighbors,
            dim=-1,
            largest=False,
        )
        weights = torch.reciprocal(torch.clamp(nearest_dist, min=1e-6) ** 2)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)

        expanded_values = values[:, :, None, :].expand(-1, -1, target_points.shape[1], -1)
        expanded_idx = nearest_idx[:, None, :, :].expand(-1, values.shape[1], -1, -1)
        gathered = torch.gather(expanded_values, dim=-1, index=expanded_idx)
        return torch.sum(gathered * weights[:, None, :, :], dim=-1)


def loss_toeplitz(A: torch.Tensor) -> torch.Tensor:
    """Differentiable penalty for the distance of a matrix from Toeplitz form."""
    if not isinstance(A, torch.Tensor):
        raise TypeError("A deve essere un torch.Tensor.")
    if A.ndim < 2:
        raise ValueError("A deve avere almeno due dimensioni.")
    if not (A.is_floating_point() or A.is_complex()):
        raise TypeError("A deve avere dtype floating point o complesso.")

    n_righe, n_colonne = A.shape[-2:]
    if n_righe == 0 or n_colonne == 0:
        raise ValueError("A deve avere righe e colonne non vuote.")

    dtype_loss = A.real.dtype if A.is_complex() else A.dtype
    scarto_quadratico = torch.zeros((), device=A.device, dtype=dtype_loss)
    numero_elementi = 0

    for k in range(1 - n_righe, n_colonne):
        diagonale = torch.diagonal(A, offset=k, dim1=-2, dim2=-1)
        media_diagonale = diagonale.mean(dim=-1, keepdim=True)
        scarto_quadratico = scarto_quadratico + torch.sum(
            torch.abs(diagonale - media_diagonale) ** 2
        )
        numero_elementi += diagonale.numel()

    return scarto_quadratico / numero_elementi


class PINN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers deve essere almeno 1.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout deve essere in [0, 1).")

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def physics_loss(
        self,
        collocation_points: torch.Tensor,
        operatore_differenziale,
        fisica,
    ) -> torch.Tensor:
        """PINN physics loss on collocation points."""
        if not callable(operatore_differenziale):
            raise TypeError("operatore_differenziale deve essere callable.")
        if fisica is None:
            raise ValueError("fisica deve essere callable, un tensore o uno scalare.")

        collocation_points = collocation_points.detach().clone().requires_grad_(True)
        u = self.forward(collocation_points)
        termine_differenziale = operatore_differenziale(u, collocation_points)

        if callable(fisica):
            residual = fisica(termine_differenziale, u, collocation_points)
        else:
            if isinstance(fisica, torch.Tensor):
                termine_fisico = fisica.to(device=u.device, dtype=u.dtype)
            else:
                termine_fisico = torch.as_tensor(fisica, device=u.device, dtype=u.dtype)
            residual = termine_differenziale - termine_fisico

        if not isinstance(residual, torch.Tensor):
            residual = torch.as_tensor(residual, device=u.device, dtype=u.dtype)

        return torch.mean(residual ** 2)

    def data_loss(self, x_data: torch.Tensor, y_data: torch.Tensor) -> torch.Tensor:
        y_pred = self.forward(x_data)
        return torch.mean((y_pred - y_data) ** 2)

    def dmd_loss(
        self,
        x_current: torch.Tensor,
        x_next: torch.Tensor,
        dmd_operator: torch.Tensor,
    ) -> torch.Tensor:
        x_next_pred = x_current @ dmd_operator.T
        return torch.mean((x_next_pred - x_next) ** 2)

    def total_loss(
        self,
        collocation_points: torch.Tensor,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        operatore_differenziale,
        fisica,
        lambda_physics: float = 1.0,
        lambda_data: float = 1.0,
        dmd_current: torch.Tensor | None = None,
        dmd_next: torch.Tensor | None = None,
        dmd_operator: torch.Tensor | None = None,
        lambda_dmd: float = 0.0,
    ) -> torch.Tensor:
        loss_pde = self.physics_loss(collocation_points, operatore_differenziale, fisica)
        loss_data = self.data_loss(x_data, y_data)
        loss_totale = lambda_physics * loss_pde + lambda_data * loss_data

        if lambda_dmd != 0.0:
            if dmd_current is None or dmd_next is None or dmd_operator is None:
                raise ValueError(
                    "dmd_current, dmd_next e dmd_operator sono necessari "
                    "quando lambda_dmd != 0."
                )
            loss_totale = loss_totale + lambda_dmd * self.dmd_loss(
                dmd_current,
                dmd_next,
                dmd_operator,
            )

        return loss_totale
