# -Privacy-Preserving-Federated-Learning-for-Alzheimer-s-Classification
A truly private federated learning system for Alzheimer's disease classification that combines differential privacy with genuine homomorphic encryption. Unlike traditional implementations, this system guarantees that the central server never sees individual client gradients in plaintext.
🔒 Advanced Privacy Guarantees
What Makes This Implementation Superior?
AspectTraditional ApproachesOur Advanced ImplementationServer sees individual gradients❌ YES (Privacy Risk)✅ NO (Fully Protected)Computation location❌ Plaintext processing✅ Encrypted space computationPrivacy guarantee❌ Limited guarantees✅ Mathematical proofCryptographic security❌ Basic encryption✅ IND-CPA secure HEMedical data protection❌ Standard security✅ HIPAA-compliant
🛡️ Privacy Mechanisms

Differential Privacy (DP): Adds calibrated Gaussian noise to gradients

(ε,δ)-differential privacy with mathematically proven bounds
Configurable privacy budget management
Gradient clipping for bounded sensitivity


Advanced Homomorphic Encryption (HE): Using TenSEAL/CKKS scheme

Server performs computations on encrypted data only
Individual client gradients never decrypted at server
Public key encryption ensures client-side privacy



🏥 Medical Use Case: Federated Alzheimer's Classification
This system enables multiple hospitals to collaboratively train an Alzheimer's classification model without sharing sensitive patient data:

Binary Classification: Nondemented vs Demented patients
7 Key Features: Age, Education, Socioeconomic Status, Brain Volume metrics
Enhanced Preprocessing: Based on proven KNN analysis approach
Federated Architecture: 3+ hospital simulation with realistic data distribution

🚀 Quick Start
Prerequisites
bash# Install required packages
pip install torch pandas scikit-learn matplotlib tenseal numpy

# Or install from requirements.txt
pip install -r requirements.txt
Setup Verification
bash# Check if everything is properly installed
python setup_checker.py
Basic Usage
python# Run the complete federated learning experiment
python real_main_server.py
This will:

🏥 Initialize 3 hospital clients with distributed data
🔒 Set up differential privacy + homomorphic encryption
🔄 Run 5 federated learning rounds
📊 Generate comprehensive privacy and performance plots
🛡️ Guarantee server never sees individual hospital gradients!

📁 Project Structure
├── real_main_server.py           # Main federated learning coordinator
├── federated_clients.py          # Hospital client simulator
├── real_privacy_utils.py         # Privacy mechanisms (DP + HE)
├── setup_checker.py              # Installation verification
├── privacy_comparison.py          # Privacy implementation comparison
├── alzheimer.csv                 # Dataset (required)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
📋 File Descriptions
real_main_server.py

Main coordinator for privacy-preserving federated learning
Orchestrates communication between hospital clients
Implements secure aggregation with true privacy guarantees
Generates comprehensive evaluation plots and privacy metrics

federated_clients.py

Hospital client simulator with realistic data distribution
Implements improved neural network architecture for Alzheimer's classification
Uses proven preprocessing approach (NaN removal, binary classification, feature selection)
Handles local training with early stopping and gradient clipping

real_privacy_utils.py

Core privacy mechanisms:

DifferentialPrivacy: Gaussian noise mechanism with (ε,δ)-DP
AdvancedHomomorphicEncryption: Secure HE using TenSEAL/CKKS
SecureAggregator: Combines DP + HE for complete privacy
PrivacyAccountant: Tracks privacy budget consumption



privacy_comparison.py

Educational demonstration showing the critical difference between:

Traditional HE: Server decrypts and processes in plaintext
Advanced HE: Server computes on encrypted data exclusively



🔧 Configuration Options
Privacy Parameters
python# In real_main_server.py
PRIVACY_EPSILON = 2.0      # Differential privacy parameter (smaller = more private)
NUM_CLIENTS = 3            # Number of hospital clients
NUM_ROUNDS = 5             # Federated learning rounds
LOCAL_EPOCHS = 100         # Training epochs per client per round
Model Architecture
python# Neural network configuration
INPUT_SIZE = 7             # Features: Age, EDUC, SES, eTIV, nWBV, ASF, M/F
HIDDEN_SIZE = 64           # Hidden layer neurons
DROPOUT_RATE = 0.2         # Regularization
Homomorphic Encryption
python# HE security parameters
POLY_MODULUS_DEGREE = 8192     # Security level (higher = more secure)
COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]  # Coefficient modulus chain
SCALE = 2.0**40                # Encoding precision
📊 Results and Visualization
The system generates comprehensive plots showing:

Global Model Accuracy: Performance improvement over federated rounds
Individual Client Performance: Each hospital's model accuracy
Privacy Budget Consumption: Differential privacy budget usage
Processing Time: Including homomorphic encryption overhead
Homomorphic Operations: Count of encrypted computations
Privacy vs Utility Trade-off: Accuracy vs privacy budget relationship

🔬 Technical Details
Privacy Analysis

Differential Privacy: (ε,δ)-DP with ε=2.0, δ=1e-5
Homomorphic Encryption: CKKS scheme with 128-bit security
Gradient Clipping: L2 norm bounded to 1.0 for sensitivity control
Noise Calibration: Gaussian mechanism with σ = √(2ln(1.25/δ)) * S / ε

Security Guarantees

Individual Privacy: No single hospital's data can be reconstructed
Aggregate Utility: Global model maintains high accuracy
Cryptographic Security: IND-CPA secure under RLWE assumption
Composition Privacy: Privacy budget tracked across multiple rounds

🏥 Medical Data Preprocessing
Dataset Features (Post-processing)

Age: Patient age in years
EDUC: Years of education
SES: Socioeconomic status (1-5 scale)
eTIV: Estimated total intracranial volume
nWBV: Normalized whole brain volume
ASF: Atlas scaling factor
M/F: Gender (binary encoded)

Preprocessing Pipeline

NaN Removal: Complete case analysis (proven approach)
Class Filtering: Remove "Converted" class for binary classification
Feature Selection: Remove CDR and MMSE (based on KNN analysis)
Standardization: Z-score normalization per client
Stratified Splitting: Balanced train/test distribution

🔍 Privacy Verification
Run the Comparison Demo
bashpython privacy_comparison.py
This demonstrates the crucial difference between traditional and advanced homomorphic encryption implementations.
Privacy Audit Checklist

✅ Server Blindness: Server never sees plaintext gradients
✅ Encryption Verification: All computations in encrypted space
✅ Differential Privacy: Calibrated noise addition
✅ Budget Tracking: Privacy budget consumption monitored
✅ Gradient Clipping: Sensitivity bounds enforced
✅ Mathematical Proof: (ε,δ)-DP guarantees maintained

🐛 Troubleshooting
Common Issues

TenSEAL Installation Fails
bash# Try conda instead of pip
conda install -c conda-forge tenseal

CUDA/Memory Issues
python# Reduce batch size or model size
HIDDEN_SIZE = 32  # Instead of 64

Dataset Not Found
bash# Ensure alzheimer.csv is in the project root
ls alzheimer.csv

Import Errors
bash# Verify all files are present
python setup_checker.py


📚 Research References
This implementation is based on established research in:

Differential Privacy: Dwork, C. (2006). "Differential Privacy"
Federated Learning: McMahan, B. et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data"
Homomorphic Encryption: Cheon, J.H. et al. (2017). "Homomorphic Encryption for Arithmetic of Approximate Numbers"
Privacy-Preserving ML: Geyer, R.C. et al. (2017). "Differentially Private Federated Learning"

🤝 Contributing

Fork the repository
Create a feature branch (git checkout -b feature/privacy-enhancement)
Commit your changes (git commit -am 'Add new privacy feature')
Push to the branch (git push origin feature/privacy-enhancement)
Create a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

TenSEAL Team for providing true homomorphic encryption capabilities
OpenMined for privacy-preserving ML tools
Medical AI Research Community for federated learning in healthcare

⚠️ Disclaimer
This is a research implementation. For production medical applications, additional security audits, compliance verification, and clinical validation are required.

🛡️ Remember: Advanced privacy isn't just about encryption - it's about never decrypting individual data at the server. This implementation guarantees mathematical privacy with rigorous cryptographic foundations!

Dr.Mohammed Tawfik 
Kmkhol01@gmail.com
