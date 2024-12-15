import torch
import torch.nn as nn
from torch.nn.functional import relu


class EEGtoMEGUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder 
        # First Level - Modified to preserve temporal dimension better
        self.e11 = nn.Conv2d(1, 128, kernel_size=(77, 3), padding=(0, 1), stride=(1, 1))    
        self.e12 = nn.Conv2d(128, 128, kernel_size=(1, 3), padding=(0, 1))   
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))         

        # Second Level
        self.e21 = nn.Conv2d(128, 256, kernel_size=(1, 3), padding=(0, 1))   
        self.e22 = nn.Conv2d(256, 256, kernel_size=(1, 3), padding=(0, 1))   
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Third Level
        self.e31 = nn.Conv2d(256, 512, kernel_size=(1, 3), padding=(0, 1))   
        self.e32 = nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1))   
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))        

        # Bridge - Adjusted sizes based on actual tensor shapes
        self.flatten = nn.Flatten()
        self.bridge_mlp = nn.Sequential(
            nn.Linear(17408, 4096),  # 17408 = 512 * 1 * 34
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 17408),
        )
        self.unflatten = nn.Unflatten(1, (512, 1, 34))

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 2), stride=(1, 2))
        self.d11 = nn.Conv2d(1024, 512, kernel_size=(1, 3), padding=(0, 1))   
        self.d12 = nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1))    

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=(1, 2), stride=(1, 2))
        self.d21 = nn.Conv2d(512, 256, kernel_size=(1, 3), padding=(0, 1))    
        self.d22 = nn.Conv2d(256, 256, kernel_size=(1, 3), padding=(0, 1))    

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=(1, 2), stride=(1, 2))
        self.d31 = nn.Conv2d(256, 128, kernel_size=(1, 3), padding=(0, 1))    
        self.d32 = nn.Conv2d(128, 128, kernel_size=(1, 3), padding=(0, 1))    

        # Output layer
        self.outconv = nn.Conv2d(128, 102, kernel_size=(1, 1))                   
        
        # Final upsampling to match target size
        self.final_upsample = nn.Upsample(size=(1, 275), mode='bilinear', align_corners=False)
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()


    def forward(self, x):
        # Reshape input: [batch, channels, time] -> [batch, 1, channels, time]
        x = x.unsqueeze(1)
        
        # Encoder
        xe11 = self.relu(self.e11(x))
        xe12 = self.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)
        xp1 = self.dropout(xp1)

        xe21 = self.relu(self.e21(xp1))
        xe22 = self.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)
        xp2 = self.dropout(xp2)

        xe31 = self.relu(self.e31(xp2))
        xe32 = self.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)
        xp3 = self.dropout(xp3)

        # Bridge
        xb = self.flatten(xp3)
        xb = self.bridge_mlp(xb)
        xb2 = self.unflatten(xb)
        xb2 = self.dropout(xb2)


        # Decoder with size matching
        xu1 = self.upconv1(xb2)
        xu1 = torch.nn.functional.pad(xu1, [0, xe32.size(-1) - xu1.size(-1), 0, 0])
        xu11 = torch.cat([xu1, xe32], dim=1)
        xd11 = self.relu(self.d11(xu11))
        xd12 = self.relu(self.d12(xd11))
        xd12 = self.dropout(xd12)

        xu2 = self.upconv2(xd12)
        xu2 = torch.nn.functional.pad(xu2, [0, xe22.size(-1) - xu2.size(-1), 0, 0])
        xu22 = torch.cat([xu2, xe22], dim=1)
        xd21 = self.relu(self.d21(xu22))
        xd22 = self.relu(self.d22(xd21))
        xd22 = self.dropout(xd22)

        xu3 = self.upconv3(xd22)
        xu3 = torch.nn.functional.pad(xu3, [0, xe12.size(-1) - xu3.size(-1), 0, 0])
        xu33 = torch.cat([xu3, xe12], dim=1)
        xd31 = self.relu(self.d31(xu33))
        xd32 = self.relu(self.d32(xd31))

        # Output layer
        out = self.outconv(xd32)
        
        # Upsample to match target size
        out = self.final_upsample(out)
        
        # Remove the singleton dimension
        out = out.squeeze(2)
        
        return out