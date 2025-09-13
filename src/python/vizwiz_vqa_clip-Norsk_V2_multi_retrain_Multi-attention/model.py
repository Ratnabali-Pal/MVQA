import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, embedding_size, classes, aux_classes):
        super(Model, self).__init__()
        self.ln1 = nn.LayerNorm(embedding_size * 2)
        self.dp1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embedding_size * 2, 512)
        
        self.ln2 = torch.nn.LayerNorm(512)
        self.dp2 = torch.nn.Dropout(0.5)
        
        self.fc2 = torch.nn.Linear(512, classes)
        
        self.fc_aux = torch.nn.Linear(512, aux_classes)
        
        self.lnaux = torch.nn.LayerNorm(aux_classes)
        self.dpaux = torch.nn.Dropout(0.5)
                
        self.fc_gate = torch.nn.Linear(aux_classes, classes)
        self.act_gate = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.ln1(x)
        x = self.dp1(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        x = self.ln2(x)
        x = self.dp2(x)
        
        # Auxiliary branch
        aux = self.fc_aux(x)
        
        # VQA branch
        vqa = self.fc2(x)
        
        output = vqa * self.act_gate(self.fc_gate(aux))
        
        return output, aux