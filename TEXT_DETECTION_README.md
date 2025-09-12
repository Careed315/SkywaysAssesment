# AI Text-Based Object Detection

This tool now supports **state-of-the-art AI object detection** using natural language prompts powered by GroundingDINO.

## Features

### 🤖 **AI-Powered Detection**
- Detect objects using natural language descriptions
- Examples: "person with red shirt", "blue car", "dog", "bicycle"
- State-of-the-art GroundingDINO model
- GPU acceleration when available

### 🔄 **Intelligent Workflow**
1. **AI Detection First**: Use `--text-prompt` to enable AI detection
2. **Tracking Takes Over**: After first detection, tracker follows the object
3. **Smart Fallbacks**: Multiple options when tracking fails

## Installation

### Basic Installation (Manual Labeling Only)
```bash
pip install -r requirements.txt
```

### Full Installation (AI Detection + Manual)
```bash
# Install basic requirements
pip install -r requirements.txt

# Install AI detection dependencies
pip install torch torchvision transformers groundingdino-py supervision
```

## Usage

### Manual Labeling (Original Mode)
```bash
python main.py video.mp4
```

### AI Text Detection Mode
```bash
# Detect specific objects
python main.py video.mp4 --prompt "person with red shirt"
python main.py video.mp4 --prompt "blue car"
python main.py video.mp4 --prompt "dog running"

# With custom annotations file
python main.py video.mp4 --prompt "bicycle" --annotations bikes.txt
```

## Workflow

### 🎯 **AI Detection Mode Workflow**
*Activated when you use `--prompt "description"`*

#### **No Prediction Available:**
- **L/l** → Run AI text detection
- **S/s** → Skip frame  
- **I/i** → Mark as invisible
- **Q/q** → Quit

#### **Prediction Available (from tracker):**
- **A/a** → Accept prediction (if tracking is good)
- **L/l** → Re-run AI detection (if tracking failed)
- **F/f** → Manual fix (as last resort)
- **S/s** → Skip frame
- **I/i** → Mark as invisible  
- **Q/q** → Quit

### 📋 **Decision Tree**
```
Frame N
├── Has tracker prediction?
│   ├── YES → Tracker did good job?
│   │   ├── YES → Press A/a (Accept)
│   │   └── NO → Press L/l (Re-detect) or F/f (Manual fix)
│   └── NO → Press L/l (Run AI detection)
```

## AI Detection Examples

### Good Prompts:
- ✅ `"person"` - General object detection
- ✅ `"person with red shirt"` - Specific attributes
- ✅ `"blue car"` - Color + object
- ✅ `"dog running"` - Object + action
- ✅ `"bicycle on the street"` - Object + context

### Tips for Better Detection:
- Be specific but not overly complex
- Include colors, sizes, or distinctive features
- Use common object names
- Describe what makes the object unique

## Technical Details

### Model Information
- **Model**: GroundingDINO (State-of-the-art open-source)
- **Capability**: Natural language object detection
- **Device**: Automatic GPU/CPU selection
- **Performance**: Real-time detection on modern hardware

### File Structure
```
├── main.py              # Main annotation tool
├── text_detector.py     # AI detection module (separate file)
├── requirements.txt     # Dependencies
└── README.md           # Documentation
```

### Configuration
- **Confidence Threshold**: 0.35 (adjustable)
- **Device**: Auto-detects CUDA/CPU
- **Model Download**: Automatic on first use

## Troubleshooting

### AI Detection Not Working?
1. **Check Installation**:
   ```bash
   pip install torch torchvision transformers groundingdino-py supervision
   ```

2. **Verify Dependencies**:
   ```python
   python -c "import torch; print('PyTorch:', torch.__version__)"
   ```

3. **Check Console Output**: Look for detection status messages

### Performance Issues?
- **GPU Memory**: Use smaller batch sizes
- **CPU Mode**: Detection will be slower but functional
- **Model Download**: First run downloads model files

### No Detections Found?
- Try different prompt phrasings
- Adjust confidence threshold
- Ensure object is clearly visible
- Use more specific descriptions

## Examples in Action

```bash
# Detect people in a crowd
python main.py crowd.mp4 --prompt "person"

# Track a specific vehicle
python main.py traffic.mp4 --prompt "red truck"

# Follow an animal
python main.py nature.mp4 --prompt "bird flying"

# Sports tracking
python main.py game.mp4 --prompt "player with number 10"
```

This AI-powered detection makes annotation **significantly faster** and more accurate compared to manual labeling alone!
