from dataset.dataSet import OpenFMRIDataSet
from dotenv import load_dotenv
import os
from torch.utils.data import DataLoader
from models.models import EEGtoMEGUNet
import matplotlib.pyplot as plt
import json
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
import traceback
import signal
import sys
import os
import time
import wandb


# Flag to indicate if termination has been requested
termination_requested = False

def signal_handler(signum, frame):
    global termination_requested    
    print(f"Received signal {signum}. Termination requested.")
    termination_requested = True
    raise KeyboardInterrupt 


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load training runs configuration
    with open('training_runs.json', 'r') as f:
        training_runs = json.load(f)

    for run_config in training_runs:
        run_name = run_config.get('name', 'default_run')
        num_epochs = run_config.get('epochs', 10)
        files_percentage = run_config.get('files_percentage', 1.0)
        verbose = run_config.get('verbose', False)
        num_workers = run_config.get('num_workers', 4)
        batch_size = run_config.get('batch_size', 128)
        prefetch_factor = run_config.get('prefetch_factor', 3)
        model_weights_file = run_config.get('model_weights_file', '')
        learning_rate = run_config.get('learning_rate', 0.0001)
        task_mode = run_config.get('task_mode', 'gait')
        breakRun = run_config.get('break', False)

        if breakRun:
            break
        
        # Initialize W&B run
        wandb.init(project="Synaptech", 
                   name=run_name, 
                   config=run_config)

        log_folder_name = f'runs/{run_name}'
        os.makedirs(log_folder_name, exist_ok=True)

        weight_file_set = False
        initialWeightsFile = f"{log_folder_name}/model_initial.pth"
        if model_weights_file:
            initialWeightsFile = model_weights_file
            weight_file_set = True
        
        def printConfig():
            logger.info("INITIALIZING TRAINING RUN")
            logger.info("Run Configuration:")
            logger.info(f"run_name: {run_name}")
            logger.info(f"num_epochs: {num_epochs}")
            logger.info(f"learning_rate: {learning_rate}")
            logger.info(f"model_weights_file: {model_weights_file}")
            logger.info(f"files_percentage: {files_percentage}")
            logger.info(f"verbose: {verbose}")
            logger.info(f"batch_size: {batch_size}")
            logger.info(f"num_workers: {num_workers}")
            logger.info(f"prefetch_factor: {prefetch_factor}")
            
        logging.basicConfig(filename=f'{log_folder_name}/training_log.log', level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        
        logger = logging.getLogger()
        
        ### suppresses annoying useless warnings
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

        if verbose:
            logger.setLevel(logging.DEBUG)

        ## print config
        printConfig()

        logger.info("Checking device..")
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using: {device}") 

        logger.info("Loading Data...")

        train_loader = None
        #test_loader = None
        val_loader = None

        logger.info("Loading OpenFMRI dataset...")
        
        load_dotenv()
        dataset_path = os.getenv("DATASET_PATH")


        ### init dataset this will automatically download and preprocess the dataset if run for the first time
        # Corrected parameters based on the new constructor
        train_dataset = OpenFMRIDataSet(
            datasetPath=dataset_path,
            mode='train',
            windowLength_ms=250,  # new parameter name
            windowOverlap=0.3,     # Adjust if needed
            logger=logger          # Optional, remove if not using a custom logger
        )

        val_dataset = OpenFMRIDataSet(
            datasetPath=dataset_path,
            mode='val',
            windowLength_ms=250,  # new parameter name
            windowOverlap=0.3,     # Adjust if needed
            logger=logger          # Optional
        )

        test_dataset = OpenFMRIDataSet(
            datasetPath=dataset_path,
            mode='test',
            windowLength_ms=250,  # new parameter name
            windowOverlap=0.3,     # Adjust if needed
            logger=logger          # Optional
        )
                
        # Set num_workers=0 to avoid issues with HDF5 file access in multiple workers
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=0,   # Set to 0 for HDF5 single-threaded access
            pin_memory=True if device.type == 'cuda' else False
        )

        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True if device.type == 'cuda' else False
        )

        #INIT MODEL
        logger.info("Initializing model...")
        
        model = EEGtoMEGUNet()
        
        model = model.to(device)

        logger.info("Initializing loss function & optimizer...")
        
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(params=model.parameters())

        epoch_stats = None

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        train_f1_scores = []
        val_f1_scores = []

        y_true = []
        y_scores = []

        ### Load initial weights file
        if not os.path.isfile(initialWeightsFile) and weight_file_set:
            logger.error(f"Provided model weights file {initialWeightsFile} does not exist. Exiting.")
            sys.exit(1)
        elif not os.path.isfile(initialWeightsFile):
            logger.warning(f"No initial model weights file {initialWeightsFile} found. Continuing training from scratch.")
            torch.save(model.state_dict(), initialWeightsFile)
            logger.info(f"Initial model weights saved to {initialWeightsFile}")
        else:
            logger.info(f"Loading model weights from {initialWeightsFile}")
            model.load_state_dict(torch.load(initialWeightsFile, map_location=device))
            logger.info(f"Model initialized with weights from {initialWeightsFile}")

        logger.info(f"Training for {num_epochs} epochs...")
        try:
            for epoch in range(num_epochs):
                if termination_requested:
                    logger.warning("Termination requested. Exiting outer training loop.")
                    break

                # TRAINING 
                running_loss = 0.0
                correct = 0
                total = 0
                index = 0
                
                model.train()  
                
                progress_bar = tqdm(total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}] - Training - started {time.strftime('%H:%M')}", leave=True)
                
                y_true = []
                y_pred = []
                
                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    if termination_requested:
                        logger.warning("Termination requested. Exiting inner training loop.")
                        break
                    
                    progress_bar.update(1)
                                        
                    inputs = inputs.float().to(device)  # Add .float() to convert to float32
                    labels = labels.float().to(device)

                    index += 1
            
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    # Track training loss
                    running_loss += loss.item()
         

                    # Log batch metrics to W&B
                    wandb.log({
                        'batch_idx': batch_idx,
                        'batch_loss': loss.item(),
                    })

                ### calculate epoch's training loss
                train_loss = running_loss / len(train_loader)
                train_losses.append(train_loss)
                
                
                # VALIDATION 
                model.eval()  # Set the model to evaluation mode
                val_running_loss = 0.0
                
                
                with torch.no_grad():  # Disable gradient computation for validation
                    progress_bar = tqdm(total=len(val_loader), desc=f"Epoch [{epoch+1}/{num_epochs}] - Validation - started {time.strftime('%H:%M')}", leave=False)
                    index = 0

                    for inputs, labels in val_loader:  
                        if termination_requested:
                            logger.warning("Termination requested. Exiting validation loop.")
                            break
                        index += 1
                        
                        inputs = inputs.float().to(device)
                        labels = labels.float().to(device)

                        outputs = model(inputs)

                        progress_bar.update(1)

                        loss = criterion(outputs, labels)
                        val_running_loss += loss.item()
            
                    progress_bar.close()

                ### calculate epoch's validation loss, accuracy, and F1 score
                val_loss = val_running_loss / len(val_loader)
                
                val_losses.append(val_loss)
                

                # Log epoch metrics to W&B
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })

                # Log and save training history and weights
                logger.info(f"EPOCH {epoch+1}: Val Loss: {val_loss:.4f}")
                
                if num_epochs > 20:
                    if (epoch + 1) % 4 == 0:
                        torch.save(model.state_dict(), f'{log_folder_name}/model_epoch_{epoch+1}.pth')
                elif num_epochs > 15:
                    if (epoch + 1) % 3 == 0:
                        torch.save(model.state_dict(), f'{log_folder_name}/model_epoch_{epoch+1}.pth')
                elif num_epochs > 10:
                    if (epoch + 1) % 2 == 0:
                        torch.save(model.state_dict(), f'{log_folder_name}/model_epoch_{epoch+1}.pth')
                else:
                    torch.save(model.state_dict(), f'{log_folder_name}/model_epoch_{epoch+1}.pth')

                with open(f'{log_folder_name}/training_history.json', 'w') as f:
                    json.dump({
                        'train_losses': train_losses,
                        'val_losses': val_losses
                    }, f)

            logger.info(f"TRAINING {num_epochs} EPOCHS COMPLETED")
        except Exception as e:
            torch.save(model.state_dict(), f'{log_folder_name}/model_epoch_{epoch+1}_error.pth')
            logger.error(f"Training stopped due to an error: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            wandb.finish()
            logger.info("Saving epoch statistics")
            if epoch_stats is not None:
                with open(os.path.join(log_folder_name, 'epoch_stats.json'), 'w') as f:
                    json.dump(epoch_stats, f, indent=4)

if __name__ == "__main__":
    main()
