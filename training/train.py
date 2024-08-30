import logging

def train(self, epochs=1):
    self.model.train()
    round_losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(self.data_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.loss_fn(output, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            logging.debug(f"[{self.addr}:{self.port}] Batch {batch_idx + 1}/{len(self.data_loader)}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(self.data_loader)
        round_losses.append(avg_loss)
        logging.debug(f"[{self.addr}:{self.port}] Average loss for the epoch: {avg_loss:.6f}")

    round_avg_loss = sum(round_losses) / len(round_losses)
    
    return round_avg_loss

    