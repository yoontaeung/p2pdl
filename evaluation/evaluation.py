import logging
import torch

def evaluate(self):
    logging.debug(f"[{self.addr}:{self.port}] Evaluating aggregated model ...")
    self.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in self.data_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    logging.info(f"[{self.addr}:{self.port}] Model accuracy: {accuracy:.2f}%")