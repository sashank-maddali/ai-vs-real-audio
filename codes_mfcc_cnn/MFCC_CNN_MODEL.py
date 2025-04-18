# Updated Part 2: Model Training with Early Stopping and Enhanced Plot Saving

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from tqdm import tqdm

# Configuration
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOPPING_PATIENCE = 5  # Number of epochs to wait for improvement before stopping

class CNNClassifier(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        
        # Calculate feature dimensions based on input shape
        freq_dim, time_dim = input_shape
        
        # CNN layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.4)
        
        # Calculate output dimensions after pooling
        out_freq = freq_dim // 8  # Three pooling layers with kernel size 2
        out_time = time_dim // 8
        
        #add the change here -> Calculate the flattened size
        
        self.fc_input_size = 64 * out_freq * out_time
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * out_freq * out_time, 128)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        # CNN blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        # Flatten and pass through FC layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x

class EarlyStopping:
    """Early stopping to terminate training when validation loss doesn't improve."""
    def __init__(self, patience=5, min_delta=0, output_dir=None):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement
            output_dir (str): Directory to save model checkpoints
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.output_dir = output_dir or os.getcwd()
        self.best_model_path = os.path.join(self.output_dir, 'best_model.pth')
        
    def __call__(self, val_loss, model):
        score = -val_loss  # Higher score is better
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        """Save model when validation loss decreases."""
        torch.save(model.state_dict(), self.best_model_path)
        print(f'Validation loss decreased. Saving model to {self.best_model_path}')

def train_model(model, train_loader, test_loader, output_dir, feature_type, num_epochs=NUM_EPOCHS):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, output_dir=output_dir)
    
    # Training history
    train_losses = []
    test_losses = []
    test_accs = []
    test_aucs = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE) #need to add the change if change in part1 does not work
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        
        # Validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Testing"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                probabilities = F.softmax(outputs, dim=1)[:, 1]  # Probability for class 1 (AI)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_acc = correct / total
        test_losses.append(epoch_loss)
        test_accs.append(epoch_acc)
        
        # Calculate ROC-AUC
        auc = roc_auc_score(all_labels, all_probs)
        test_aucs.append(auc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Test Loss: {epoch_loss:.4f}, Test Acc: {epoch_acc:.4f}, AUC: {auc:.4f}")
        
        # Adjust learning rate
        scheduler.step(epoch_loss)
        
        # Early stopping check
        early_stopping(epoch_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Save training plots at the end of training
    plot_training_history(train_losses, test_losses, test_accs, test_aucs, feature_type, output_dir)
    
    # Load the best model for final evaluation
    model.load_state_dict(torch.load(early_stopping.best_model_path))
    
    return model, train_losses, test_losses, test_accs, test_aucs

def evaluate_model(model, test_loader, output_dir, feature_type):
    """Comprehensive model evaluation with ROC curve and metrics"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE) #add the change here below 
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            probabilities = F.softmax(outputs, dim=1)[:, 1]  # Probability for class 1 (AI)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_scores = np.array(all_probs)
    
    # Calculate metrics
    accuracy = (y_pred == y_true).mean()
    auc = roc_auc_score(y_true, y_scores)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Print results
    print("\nFinal Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {auc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Save classification report to file
    report = classification_report(y_true, y_pred, target_names=['Real', 'AI'])
    print("\nClassification Report:")
    print(report)
    
    with open(os.path.join(output_dir, f'classification_report_{feature_type}.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC-AUC Score: {auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix))
        f.write("\n\nClassification Report:\n")
        f.write(report)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Plot and save ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {feature_type.upper()} Features')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Save ROC curve
    roc_path = os.path.join(output_dir, f'roc_curve_{feature_type}.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of plt.show()
    
    print(f"ROC curve saved to {roc_path}")
    
    return accuracy, auc, y_pred, y_scores, y_true

def plot_training_history(train_losses, test_losses, test_accs, test_aucs, feature_type, output_dir):
    """Plot and save training history"""
    plt.figure(figsize=(20, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot validation accuracy
    plt.subplot(1, 3, 2)
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot AUC
    plt.subplot(1, 3, 3)
    plt.plot(test_aucs, label='AUC-ROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('ROC AUC Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    history_path = os.path.join(output_dir, f'training_history_{feature_type}.png')
    plt.savefig(history_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of plt.show()
    
    print(f"Training history saved to {history_path}")

def main():
    # Import needed functions from Part 1
    from part1_fe import prepare_dataset
    
    # Set paths to your dataset folders
    real_audio_folder = "/home/sashank/Desktop/project/wav_2vec/real_16"
    ai_audio_folder = "/home/sashank/Desktop/project/wav_2vec/ai_16"
    feature_type = "mfcc"  # or "mel"
    
    # Create output directory in the same location as the dataset
    output_dir = os.path.join(os.path.dirname(real_audio_folder), f"results_{feature_type}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Results will be saved to: {output_dir}")
    
    # Prepare data
    train_loader, test_loader, input_shape = prepare_dataset(
        real_audio_folder,
        ai_audio_folder,
        feature_type=feature_type
    )
    
    print(f"Input feature shape: {input_shape}")
    
    # Initialize model
    model = CNNClassifier(input_shape).to(DEVICE)
    print(model)
    
    # Train model
    trained_model, train_losses, test_losses, test_accs, test_aucs = train_model(
        model,
        train_loader,
        test_loader,
        output_dir,
        feature_type
    )
    
    # Evaluate model with ROC
    accuracy, auc, y_pred, y_scores, y_true = evaluate_model(
        trained_model, 
        test_loader,
        output_dir,
        feature_type
    )
    
    # Save the final model
    final_model_path = os.path.join(output_dir, f'final_model_{feature_type}.pth')
    torch.save(trained_model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save metrics for comparison with other models
    results = {
        'model_type': f'CNN_{feature_type.upper()}',
        'accuracy': accuracy,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_scores,
        'true_labels': y_true
    }
    
    # Save a simple results summary
    with open(os.path.join(output_dir, f'results_summary_{feature_type}.txt'), 'w') as f:
        f.write(f"Model: CNN with {feature_type.upper()} features\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"AUC-ROC: {auc:.4f}\n")
    
    print(f"Final results: {results['model_type']} - Accuracy: {results['accuracy']:.4f}, AUC: {results['auc']:.4f}")
    return results

if __name__ == "__main__":
    results = main()
