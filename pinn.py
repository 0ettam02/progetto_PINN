import torch
import torch.nn as nn


def loss_toeplitz(A: torch.Tensor) -> torch.Tensor:
    """
    Loss differenziabile che misura quanto A è lontana dall'essere Toeplitz.

    Per ogni diagonale j-i=k calcola la media degli elementi sulla diagonale e
    penalizza la media degli scarti quadratici da quella media. Supporta anche
    batch di matrici con shape (..., m, n).
    """
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
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

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
        """
        Loss fisica della PINN sui collocation point.

        operatore_differenziale: funzione (u, collocation_points) -> termine differenziale.
        fisica: funzione (termine_differenziale, u, collocation_points) -> residuo,
                oppure tensore/scalare da sottrarre al termine differenziale.
        """
        if not callable(operatore_differenziale):
            raise TypeError("operatore_differenziale deve essere una funzione callable.")
        if fisica is None:
            raise ValueError("fisica deve essere una funzione callable, un tensore o uno scalare.")

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
        """
        Loss sui dati osservati.
        """
        y_pred = self.forward(x_data)
        return torch.mean((y_pred - y_data) ** 2)

    def dmd_loss(self, x_current: torch.Tensor, x_next: torch.Tensor, dmd_operator: torch.Tensor) -> torch.Tensor:
        """
        Loss basata su DMD.

        x_current: stato al tempo t
        x_next: stato al tempo t+1
        dmd_operator: matrice A stimata tramite DMD, tale che x_next ≈ A x_current
        """
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
    ) -> torch.Tensor:
        """
        Loss totale combinata:
        L = λ_physics L_physics + λ_data L_data
        """
        loss_pde = self.physics_loss(collocation_points, operatore_differenziale, fisica)
        loss_data = self.data_loss(x_data, y_data)

        return (
            lambda_physics * loss_pde
            + lambda_data * loss_data
        )
