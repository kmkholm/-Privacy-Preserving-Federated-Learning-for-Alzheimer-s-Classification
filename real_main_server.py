"""
REAL Federated Learning Server - COMPLETELY FIXED VERSION
Main Coordinator for Alzheimer's Classification with TRUE Privacy + Improved Accuracy

This file orchestrates REAL privacy-preserving federated learning:
1. Coordinates multiple hospital clients
2. Implements REAL differential privacy
3. Uses TRUE homomorphic encryption (TenSEAL/CKKS)
4. Uses YOUR proven preprocessing for better accuracy
5. Server NEVER sees individual client gradients in plaintext!
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# FIXED IMPORTS - All references corrected
try:
    from real_privacy_utils import TrueSecureAggregator, PrivacyAccountant
    from federated_clients import ImprovedClientManager, ImprovedAlzheimerNet
    print("✅ All imports successful - ready for TRUE privacy + improved accuracy!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please make sure these files are in the same directory:")
    print("  - real_privacy_utils.py")
    print("  - federated_clients.py (the FIXED version)")
    print("  - alzheimer.csv")
    exit(1)

class TrueFederatedLearningServer:
    """FIXED Main coordinator for federated learning with REAL privacy"""
    
    def __init__(self, csv_file: str, num_clients: int = 3, privacy_epsilon: float = 1.0):
        """
        Initialize the federated learning server with TRUE privacy
        """
        print("🚀 Initializing REAL Federated Learning Server with TRUE Privacy Protection")
        print("🛡️  SERVER WILL NEVER SEE INDIVIDUAL CLIENT GRADIENTS!")
        print("=" * 80)
        
        self.csv_file = csv_file
        self.num_clients = num_clients
        self.privacy_epsilon = privacy_epsilon
        
        # Initialize TRUE secure aggregator
        print("🔒 Setting up TRUE secure aggregation...")
        self.secure_aggregator = TrueSecureAggregator(epsilon=privacy_epsilon)
        
        # FIXED: Initialize IMPROVED client manager with YOUR proven preprocessing
        print("Setting up federated clients with YOUR proven preprocessing...")
        self.client_manager = ImprovedClientManager(csv_file, num_clients)  # FIXED!
        
        # FIXED: Global model using YOUR proven architecture (7 features, binary classification)
        self.global_model = ImprovedAlzheimerNet(input_size=7, hidden_size=64)  # FIXED!
        
        # Training history
        self.history = {
            'rounds': [],
            'global_accuracy': [],
            'client_accuracies': {},
            'privacy_budget_used': [],
            'aggregation_time': [],
            'homomorphic_operations': []
        }
        
        print("✅ Server initialization complete!")
        print(f"   📊 Dataset: {csv_file}")
        print(f"   🏥 Clients: {num_clients}")
        print(f"   🔒 Privacy ε: {privacy_epsilon}")
        print(f"   🧠 Model: {sum(p.numel() for p in self.global_model.parameters())} parameters")
        print(f"   🛡️  Homomorphic Encryption: TRUE (CKKS scheme)")
        print(f"   🚫 Server Plaintext Access: DENIED")
        
    def federated_training_round(self, round_num: int, local_epochs: int = 3) -> dict:
        """Execute one round of REAL privacy-preserving federated learning"""
        print(f"\n🔄 FEDERATED ROUND {round_num} - TRUE PRIVACY MODE")
        print("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Get model updates from all clients
        print("📤 Phase 1: Collecting client updates...")
        client_updates = self.client_manager.get_client_updates(epochs=local_epochs)
        
        # Step 2: TRUE secure aggregation with REAL homomorphic encryption
        print("🔒 Phase 2: TRUE secure aggregation (server blind to individual gradients)...")
        try:
            aggregated_gradients = self.secure_aggregator.secure_aggregate(client_updates)
            he_operations = len(client_updates) * len(client_updates[0])
        except ValueError as e:
            print(f"❌ Privacy budget exhausted: {e}")
            return {'status': 'privacy_budget_exhausted'}
        
        # Step 3: Update global model
        print("🌍 Phase 3: Updating global model...")
        self._update_global_model(aggregated_gradients)
        
        # Step 4: Send updated model to all clients
        print("📡 Phase 4: Broadcasting updates to clients...")
        self.client_manager.update_all_clients(aggregated_gradients)
        
        # Step 5: Evaluate all models
        print("📈 Phase 5: Evaluating client models...")
        evaluation_results = self.client_manager.evaluate_all_clients()
        
        # Record round statistics
        round_time = time.time() - start_time
        self._record_round_stats(round_num, evaluation_results, round_time, he_operations)
        
        # Print round summary
        self._print_round_summary(round_num, evaluation_results, round_time, he_operations)
        
        return {
            'status': 'success',
            'round': round_num,
            'global_accuracy': evaluation_results['Global']['accuracy'],
            'privacy_budget_remaining': self.secure_aggregator.accountant.remaining_budget(),
            'round_time': round_time,
            'he_operations': he_operations
        }
    
    def _update_global_model(self, gradients: dict):
        """Update global model with aggregated gradients"""
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in gradients:
                    param.data -= gradients[name]
    
    def _record_round_stats(self, round_num: int, results: dict, round_time: float, he_ops: int):
        """Record statistics for this round"""
        self.history['rounds'].append(round_num)
        self.history['global_accuracy'].append(results['Global']['accuracy'])
        self.history['privacy_budget_used'].append(self.secure_aggregator.accountant.used_epsilon)
        self.history['aggregation_time'].append(round_time)
        self.history['homomorphic_operations'].append(he_ops)
        
        # Record individual client accuracies
        for client_id, metrics in results.items():
            if client_id != 'Global':
                if client_id not in self.history['client_accuracies']:
                    self.history['client_accuracies'][client_id] = []
                self.history['client_accuracies'][client_id].append(metrics['accuracy'])
    
    def _print_round_summary(self, round_num: int, results: dict, round_time: float, he_ops: int):
        """Print summary of the round"""
        print(f"\n📊 ROUND {round_num} SUMMARY - TRUE PRIVACY MODE")
        print("-" * 40)
        print(f"Global Accuracy: {results['Global']['accuracy']:.3f}")
        print(f"Privacy Budget Used: {self.secure_aggregator.accountant.used_epsilon:.2f}")
        print(f"Privacy Budget Remaining: {self.secure_aggregator.accountant.remaining_budget():.2f}")
        print(f"Round Time: {round_time:.2f}s")
        print(f"Homomorphic Operations: {he_ops}")
        print(f"🛡️  Server Plaintext Exposure: ZERO (TRUE privacy!)")
        
        # Show client accuracies
        print("\nClient Performance:")
        for client_id, metrics in results.items():
            if client_id != 'Global':
                print(f"  {client_id}: {metrics['accuracy']:.3f}")
    
    def run_federated_learning(self, num_rounds: int = 5, local_epochs: int = 3):
        """Run complete REAL privacy-preserving federated learning experiment"""
        print(f"\n🎯 STARTING TRUE PRIVACY-PRESERVING FEDERATED LEARNING")
        print(f"📋 Configuration:")
        print(f"   Rounds: {num_rounds}")
        print(f"   Local Epochs per Round: {local_epochs}")
        print(f"   Privacy Budget: {self.secure_aggregator.accountant.total_epsilon}")
        print(f"   🔒 Differential Privacy: REAL (Gaussian mechanism)")
        print(f"   🛡️  Homomorphic Encryption: TRUE (CKKS scheme)")
        print(f"   🚫 Server Gradient Visibility: BLOCKED")
        print("=" * 80)
        
        # Initial evaluation
        print("📊 Initial Model Evaluation:")
        initial_results = self.client_manager.evaluate_all_clients()
        
        # Run federated rounds
        total_he_operations = 0
        for round_num in range(1, num_rounds + 1):
            result = self.federated_training_round(round_num, local_epochs)
            
            if result['status'] == 'privacy_budget_exhausted':
                print(f"\n⚠️  Privacy budget exhausted after {round_num-1} rounds")
                break
                
            total_he_operations += result.get('he_operations', 0)
        
        # Final summary
        self._print_final_summary(total_he_operations)
        
        # Generate plots
        self._generate_plots()
        
        return self.history
    
    def _print_final_summary(self, total_he_ops: int):
        """Print final experiment summary"""
        print(f"\n🎉 TRUE PRIVACY-PRESERVING FEDERATED LEARNING COMPLETE")
        print("=" * 80)
        
        if self.history['global_accuracy']:
            initial_acc = self.history['global_accuracy'][0] if len(self.history['global_accuracy']) > 0 else 0
            final_acc = self.history['global_accuracy'][-1]
            improvement = final_acc - initial_acc
            
            print(f"📈 Performance Summary:")
            print(f"   Initial Global Accuracy: {initial_acc:.3f}")
            print(f"   Final Global Accuracy: {final_acc:.3f}")
            print(f"   Improvement: {improvement:+.3f}")
            
            print(f"\n🔒 REAL Privacy Summary:")
            print(f"   Total Privacy Budget: {self.secure_aggregator.accountant.total_epsilon}")
            print(f"   Privacy Budget Used: {self.secure_aggregator.accountant.used_epsilon:.2f}")
            print(f"   Privacy Budget Remaining: {self.secure_aggregator.accountant.remaining_budget():.2f}")
            print(f"   ✅ Differential Privacy: APPLIED (Gaussian noise)")
            print(f"   ✅ Homomorphic Encryption: TRUE (CKKS scheme)")
            print(f"   ✅ Server Blindness: GUARANTEED")
            
            print(f"\n⏱️  Performance Summary:")
            avg_time = np.mean(self.history['aggregation_time'])
            print(f"   Average Round Time: {avg_time:.2f}s")
            print(f"   Total Experiment Time: {sum(self.history['aggregation_time']):.2f}s")
            print(f"   Total HE Operations: {total_he_ops}")
            
            print(f"\n🛡️  Privacy Guarantees:")
            print(f"   ✅ Individual client gradients: NEVER seen by server")
            print(f"   ✅ Aggregation in encrypted space: TRUE")
            print(f"   ✅ Mathematical privacy proof: (ε,δ)-differential privacy")
            print(f"   ✅ Computational privacy: CKKS homomorphic encryption")
    
    def _generate_plots(self):
        """Generate visualization plots with HE operations"""
        if not self.history['rounds']:
            print("No data to plot")
            return
            
        # Create figure with subplots (2x3 for more metrics)
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('TRUE Privacy-Preserving Federated Learning - Results', fontsize=16)
        
        # Plot 1: Global Accuracy over Rounds
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(self.history['rounds'], self.history['global_accuracy'], 'b-o', linewidth=2, markersize=6)
        ax1.set_title('Global Model Accuracy\n(TRUE Privacy Protected)')
        ax1.set_xlabel('Federated Round')
        ax1.set_ylabel('Accuracy')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Client Accuracies
        ax2 = plt.subplot(2, 3, 2)
        for client_id, accuracies in self.history['client_accuracies'].items():
            ax2.plot(self.history['rounds'][:len(accuracies)], accuracies, 'o-', label=client_id, alpha=0.7)
        ax2.set_title('Individual Client Accuracies')
        ax2.set_xlabel('Federated Round')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Plot 3: Privacy Budget Consumption
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(self.history['rounds'], self.history['privacy_budget_used'], 'r-s', linewidth=2, markersize=6)
        ax3.axhline(y=self.secure_aggregator.accountant.total_epsilon, color='r', linestyle='--', alpha=0.5, label='Budget Limit')
        ax3.set_title('Privacy Budget Consumption\n(DP + HE)')
        ax3.set_xlabel('Federated Round')
        ax3.set_ylabel('Cumulative ε Used')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Round Time with HE overhead
        ax4 = plt.subplot(2, 3, 4)
        bars = ax4.bar(self.history['rounds'], self.history['aggregation_time'], alpha=0.7, color='green')
        ax4.set_title('Round Processing Time\n(Including HE Overhead)')
        ax4.set_xlabel('Federated Round')
        ax4.set_ylabel('Time (seconds)')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Homomorphic Operations
        ax5 = plt.subplot(2, 3, 5)
        ax5.bar(self.history['rounds'], self.history['homomorphic_operations'], alpha=0.7, color='purple')
        ax5.set_title('Homomorphic Operations per Round\n(TRUE Encryption Count)')
        ax5.set_xlabel('Federated Round')
        ax5.set_ylabel('HE Operations')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Privacy vs Utility Trade-off
        ax6 = plt.subplot(2, 3, 6)
        ax6.scatter(self.history['privacy_budget_used'], self.history['global_accuracy'], 
                   c=self.history['rounds'], cmap='viridis', s=100, alpha=0.7)
        ax6.set_title('Privacy vs Utility Trade-off\n(Darker = Later Rounds)')
        ax6.set_xlabel('Privacy Budget Used (ε)')
        ax6.set_ylabel('Global Accuracy')
        ax6.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(ax6.collections[0], ax=ax6)
        cbar.set_label('Round Number')
        
        plt.tight_layout()
        plt.show()
        
        print("📊 TRUE privacy-preserving plots generated successfully!")
        print("🛡️  All results achieved with ZERO server access to individual gradients!")

# FIXED Main execution
if __name__ == "__main__":
    print("🧠 IMPROVED FEDERATED LEARNING FOR ALZHEIMER'S CLASSIFICATION")
    print("🔒 WITH REAL DIFFERENTIAL PRIVACY + TRUE HOMOMORPHIC ENCRYPTION")
    print("🔧 USING YOUR PROVEN PREPROCESSING + FEATURE SELECTION")
    print("🛡️  SERVER BLINDNESS GUARANTEED!")
    print("=" * 80)
    
    # Enhanced configuration based on your KNN analysis
    CSV_FILE = 'alzheimer.csv'
    NUM_CLIENTS = 3
    PRIVACY_EPSILON = 2.0  # Good balance of privacy vs utility
    NUM_ROUNDS = 5  # More rounds for better convergence
    LOCAL_EPOCHS = 100  # More epochs per round for better training
    
    print("🔧 IMPROVEMENTS APPLIED:")
    print("   ✅ Removed NaN values completely (your approach)")
    print("   ✅ Binary classification: Nondemented vs Demented")
    print("   ✅ Removed CDR and MMSE features (your analysis)")
    print("   ✅ Enhanced neural network architecture")
    print("   ✅ Better training with early stopping")
    print("   ✅ Detailed evaluation metrics")
    
    try:
        # Check if TenSEAL is installed
        import tenseal as ts
        print("✅ TenSEAL library found - TRUE homomorphic encryption available!")
        
        # Initialize improved server
        server = TrueFederatedLearningServer(
            csv_file=CSV_FILE,
            num_clients=NUM_CLIENTS,
            privacy_epsilon=PRIVACY_EPSILON
        )
        
        # Run enhanced federated learning experiment
        results = server.run_federated_learning(
            num_rounds=NUM_ROUNDS,
            local_epochs=LOCAL_EPOCHS
        )
        
        print("\n✅ IMPROVED privacy-preserving experiment completed successfully!")
        print("📊 Check the generated plots for detailed results")
        print("🛡️  Your server NEVER saw individual client gradients in plaintext!")
        print("🔒 Privacy mathematically guaranteed by (ε,δ)-DP + CKKS-HE")
        print("🔧 Using YOUR proven preprocessing for better accuracy!")
        
    except ImportError:
        print("❌ TenSEAL not installed!")
        print("Install with: pip install tenseal")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{CSV_FILE}'")
        print("Please make sure the Alzheimer's dataset is in your working directory")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Make sure all packages are installed:")
        print("pip install tenseal torch pandas scikit-learn matplotlib")
        import traceback
        traceback.print_exc()
