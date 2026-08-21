class InterpretablePINN(nn.Module):
    def __init__(self, layers, dropout_rate=0.1):
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-2)
        ])
        
        hidden_dim = layers[-2]
        self.bottleneck = PINNBottleneck(hidden_dim, dropout_rate)
        self.concept_head = nn.Linear(hidden_dim, 2)
        self.final_layer = nn.Linear(hidden_dim, layers[-1])
        
        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        for i in range(len(self.linears)):
            inputs = self.activation(self.linears[i](inputs))
        h = inputs 
        
        h_bar, k_hat, u_hat, eps, loss_indep, loss_rec = self.bottleneck(h)
        concept_preds = torch.sigmoid(self.concept_head(k_hat))
        u = self.final_layer(h_bar)
        
        return u, concept_preds, loss_indep, loss_rec
