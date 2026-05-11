import torch
import torch.nn as nn


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
