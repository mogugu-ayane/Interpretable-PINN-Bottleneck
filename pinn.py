import torch
import torch.nn as nn

class InterpretablePINN(nn.Module):
    def __init__(self, num_layers=4, hidden_dim=64):
        super().__init__()

        # ==========================================
        # [독립 네트워크 1] 메인 유동 (u_pred) 전용 뇌
        # ==========================================
        layers_u = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers_u.append(nn.Linear(hidden_dim, hidden_dim))
            layers_u.append(nn.Tanh())
        self.net_u_hidden = nn.Sequential(*layers_u)
        self.W_u = nn.Linear(hidden_dim, 1) #[cite: 1]

        layers_dyn = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers_dyn.append(nn.Linear(hidden_dim, hidden_dim))
            layers_dyn.append(nn.Tanh())
        self.net_dyn_hidden = nn.Sequential(*layers_dyn)

        # 병목 모듈 (문서 규칙 준수)[cite: 1]
        self.bottleneck = PINNBottleneck(hidden_dim=hidden_dim, bottleneck_dim=2)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)

        # 1. 메인 유동은 방해받지 않고 순수하게 물리 법칙을 학습합니다.
        h_u = self.net_u_hidden(inputs)
        u_pred = self.W_u(h_u)

        # 2. 미지 동역학은 뼈대부터 분리되어 홀로 세금(L_u_reg)을 감당합니다.
        h_dyn = self.net_dyn_hidden(inputs)
        u_dyn = self.bottleneck(h_dyn)

        # 문서에 명시된 대로 오직 u_pred와 u_dyn 2개만 뱉습니다.[cite: 1]
        return u_pred, u_dyn
