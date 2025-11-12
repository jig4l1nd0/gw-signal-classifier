# 🛰️ Gravitational Wave Signal Classifier

A deep learning application that uses a 1D U-Net to detect and isolate gravitational wave signals (chirps) from noisy time series data. Built with PyTorch and deployed as an interactive Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **Real-time Signal Detection**: Interactive web interface for gravitational wave signal analysis
- **Deep Learning Model**: 1D U-Net architecture optimized for time series segmentation
- **Synthetic Data Generation**: Tools to create realistic training datasets
- **Docker Deployment**: Production-ready containerized application
- **Educational Resources**: Comprehensive theory documentation

## 🎯 Demo

Try the live application: [Coming Soon - Deploy to Render]

### Example Results

The application processes noisy gravitational wave signals and outputs:

1. **Input Signal**: Raw noisy time series data
2. **Probability Mask**: Model's confidence in signal presence
3. **Filtered Signal**: Isolated gravitational wave signal

## 🧠 How It Works

### The Physics
Gravitational waves are ripples in spacetime created by accelerating massive objects, such as merging black holes or neutron stars. These signals create characteristic "chirp" patterns - frequency sweeps that increase in both frequency and amplitude as objects spiral together.

### The AI Approach
Our 1D U-Net neural network:
- **Learns** to distinguish signal from noise automatically
- **Localizes** signals precisely in time
- **Segments** the time series to isolate gravitational wave events
- **Generalizes** to handle various noise conditions

## 🚀 Quick Start

### Option 1: Run with Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/gw-signal-classifier.git
cd gw-signal-classifier

# Build and run with Docker
docker build -t gw-classifier .
docker run -p 8501:8501 gw-classifier
```

Open your browser to `http://localhost:8501`

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gw-signal-classifier.git
cd gw-signal-classifier

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate example data (optional)
python scripts/data_generator.py --generate-single example_signal.npy

# Run the application
streamlit run app.py
```

## 📁 Project Structure

```
gw-signal-classifier/
├── 📱 app.py                 # Streamlit web application
├── 🐳 Dockerfile             # Container configuration
├── ☁️ render.yaml            # Deployment configuration
├── 📋 requirements.txt       # Python dependencies
├── 🤖 unet_gw_model.pth      # Pre-trained model weights
├── 📊 example_signal.npy     # Sample data file
├── 
├── 📁 src/
│   ├── __init__.py
│   └── 🏗️ model.py           # U-Net architecture
├── 
├── 📁 scripts/
│   ├── 🔧 data_generator.py  # Synthetic data creation
│   └── 🎯 train.py           # Model training script
├── 
├── 📁 data/                  # Generated datasets (created on first run)
├── 
└── 📁 reading-material/
    └── 📚 gravitational-wave-theory.md  # Comprehensive theory guide
```

## 🛠️ Usage

### Web Application

1. **Launch the app**: Follow the Quick Start instructions
2. **Load data**: Either use the example signal or upload your own `.npy` file
3. **View results**: See the three-panel analysis showing input, prediction, and filtered output

### Data Requirements

Upload files must be:
- **Format**: `.npy` (NumPy array)
- **Shape**: `(2048,)` - exactly 2048 samples
- **Type**: Float32 or Float64
- **Content**: Time series representing potential gravitational wave data

### Example Data Preparation

```python
import numpy as np

# Your signal data (must be exactly 2048 samples)
my_signal = np.array([...])  # shape: (2048,)

# Optional: normalize your signal
my_signal = my_signal / np.std(my_signal)

# Save as .npy file
np.save('my_gravitational_wave_signal.npy', my_signal)
```

## 🔬 Model Architecture

### U-Net 1D Network
- **Encoder**: 5 levels of downsampling with MaxPooling
- **Decoder**: 4 levels of upsampling with skip connections
- **Features**: 64 → 128 → 256 → 512 → 1024 channels
- **Output**: Pixel-wise probability mask

### Key Design Choices
- **Skip connections**: Preserve fine temporal details
- **Multi-scale features**: Capture patterns at different time scales
- **Binary segmentation**: Output indicates signal presence/absence
- **CPU optimized**: Efficient inference without GPU requirements

## 🔧 Development

### Generate Training Data

```bash
# Generate default dataset (10,000 samples)
python scripts/data_generator.py

# Generate custom dataset
python scripts/data_generator.py --num-samples 5000 --noise-level 2.0 --output-dir custom_data

# Generate single example file
python scripts/data_generator.py --generate-single my_example.npy --noise-level 1.0
```

### Train the Model

```bash
# Train with default settings
python scripts/train.py

# Custom training (if implemented)
python scripts/train.py --epochs 100 --batch-size 32 --learning-rate 0.001
```

### Model Parameters
- **Input shape**: `(batch_size, 1, 2048)`
- **Output shape**: `(batch_size, 1, 2048)` 
- **Parameters**: ~31M trainable parameters
- **Training**: Binary Cross-Entropy with Logits loss

## 🚀 Deployment

### Deploy to Render (Free)

1. **Fork this repository** to your GitHub account
2. **Connect to Render**: 
   - Go to [render.com](https://render.com)
   - Connect your GitHub account
   - Select this repository
3. **Deploy**: Render will automatically use `render.yaml` configuration
4. **Access**: Your app will be available at `https://your-service-name.onrender.com`

### Deploy to Other Platforms

The Docker configuration works with:
- **Heroku**: Use `heroku.yml` or Docker deployment
- **AWS**: ECS, Fargate, or Elastic Beanstalk
- **Google Cloud**: Cloud Run or App Engine
- **Azure**: Container Instances or App Service

### Environment Variables (Optional)

```bash
MODEL_PATH=unet_gw_model.pth
EXAMPLE_SIGNAL_PATH=example_signal.npy
SIGNAL_LENGTH=2048
```

## 📚 Educational Resources

### Theory Documentation
- **[Gravitational Wave Theory](reading-material/gravitational-wave-theory.md)**: Comprehensive guide covering:
  - Physics of gravitational waves
  - Detection challenges
  - Machine learning approaches
  - U-Net architecture theory
  - Mathematical foundations

### Key Concepts
- **Chirp Signals**: Frequency sweeps from merging compact objects
- **Binary Inspiral**: The physics behind gravitational wave generation
- **Time Series Segmentation**: ML technique for signal localization
- **Multi-scale Feature Learning**: How U-Net captures temporal patterns

## 🛡️ Requirements

### System Requirements
- **Python**: 3.10 or higher
- **Memory**: 2GB+ RAM recommended
- **Storage**: ~500MB for model and dependencies

### Dependencies
- **PyTorch**: 2.3.1 (CPU version)
- **NumPy**: 1.26.4
- **Streamlit**: 1.35.0
- **Matplotlib**: 3.9.0
- **Scikit-learn**: 1.5.0
- **tqdm**: 4.66.4

## 🤝 Contributing

Contributions are welcome! Here are some areas for improvement:

### Enhancements
- [ ] Add real gravitational wave data support
- [ ] Implement different neural network architectures
- [ ] Add data augmentation techniques
- [ ] Optimize model for mobile deployment
- [ ] Add batch processing capabilities

### Getting Started
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit: `git commit -m 'Add feature'`
5. Push: `git push origin feature-name`
6. Submit a pull request

## 📈 Performance

### Model Metrics
- **Training**: Binary Cross-Entropy Loss
- **Inference Time**: ~50ms per signal (CPU)
- **Model Size**: ~125MB
- **Accuracy**: Depends on noise level and signal strength

### Computational Requirements
- **Training**: ~2-4 hours on modern CPU (10K samples)
- **Inference**: Real-time on any modern computer
- **Memory**: ~512MB RAM during inference

## 🔍 Troubleshooting

### Common Issues

**Model file not found**
```bash
# Download or ensure unet_gw_model.pth is present
python scripts/train.py  # Train a new model
```

**Invalid signal shape**
```python
# Reshape your data to (2048,)
signal = signal.reshape(-1)[:2048]  # Truncate if too long
signal = np.pad(signal, (0, 2048-len(signal)))  # Pad if too short
```

**Docker build fails**
```bash
# Clear Docker cache
docker system prune -a
docker build --no-cache -t gw-classifier .
```

### Performance Optimization
- Use CPU-optimized PyTorch for deployment
- Enable Streamlit caching for better performance
- Consider model quantization for smaller size

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LIGO Scientific Collaboration**: For advancing gravitational wave detection
- **PyTorch Team**: For the excellent deep learning framework
- **Streamlit Team**: For making ML app deployment simple
- **Scientific Community**: For open research in gravitational wave astronomy

## 📞 Contact

- **Author**: Josue Galindo
- **Email**: [ji.g4l1nd0.com]
- **GitHub**: [@jig4l1nd0](https://github.com/jig4l1nd0/)
- **LinkedIn**: (https://www.linkedin.com/in/josue-galindo/)

## 🔗 Related Projects

- **LIGO Open Science Center**: Real gravitational wave data
- **GWpy**: Gravitational wave data analysis in Python  
- **PyCBC**: Toolkit for gravitational wave astronomy
- **Bilby**: Bayesian inference for gravitational waves

---

### ⭐ Star this repository if it helped you!

**Built with ❤️ for the gravitational wave astronomy community**