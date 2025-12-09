import torch
from .device import DEVICE

class Camera:
    def __init__(self, fov=100, aspect_ratio=1.0):
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.reset()

    def reset(self):
        self.lookfrom = torch.tensor([0.0, 0.5, -10.0], device=DEVICE)
        self.lookat = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)
        
        angle = torch.tensor(torch.pi / 16, device=DEVICE)
        self.vup = torch.tensor([torch.sin(angle), torch.cos(angle), 0.0], device=DEVICE)
        
        theta = torch.tensor(self.fov * (torch.pi / 180.0), device=DEVICE)
        half_height = torch.tan(theta / 2.0)
        half_width = self.aspect_ratio * half_height
        
        self.cam_origin = self.lookfrom.clone()
        
        w = (self.lookfrom - self.lookat) / torch.norm(self.lookfrom - self.lookat)
        u = torch.cross(self.vup, w, dim=0)
        u = u / torch.norm(u)
        v = torch.cross(w, u, dim=0)
        
        self.cam_lower_left_corner = self.cam_origin - half_width * u - half_height * v - w
        self.cam_horizontal = 2 * half_width * u
        self.cam_vertical = 2 * half_height * v

    def serialize(self):
        params = torch.zeros(3 + 3 + 3 + 3, device=DEVICE)
        params[:3] = self.cam_origin
        params[3:6] = self.cam_lower_left_corner
        params[6:9] = self.cam_horizontal
        params[9:] = self.cam_vertical
        return params
    