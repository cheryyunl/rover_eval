# VortexBench Evaluation System

A comprehensive evaluation framework for visual reasoning models using 5 core metrics across temporal, spatial, quantitative, causal, and synthetic reasoning tasks.

## Quick Start

### 1. Setup

Configure your OpenAI credentials (choose one):

**Option A: Environment Variables (Recommended)**
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-4o"
```

**Option B: Edit config.py**
```python
OPENAI_API_KEY = "your-api-key"
OPENAI_MODEL = "gpt-4o"
```

**Legacy Azure Support (Deprecated)**
```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"
```

### 2. Prepare Generated Results

Update the generation directory path in `evaluate_vortex.py`:
```python
VORTEX_GEN_DIR = "/path/to/your/generated/results"  # Line 22
```

### 3. Generated Files Format

Your generation directory should contain files in this format:
```
your_gen_dir/
├── gen_{task_id}.png          # Generated image (required)
├── gen_{task_id}.txt          # Reasoning text (optional)
├── gen_science_temporal_1.png
├── gen_science_temporal_1.txt
├── gen_science_causal_3.png
├── gen_science_causal_3.txt
└── ...
```

**File Naming Convention:**
- Images: `gen_{task_id}.png`
- Reasoning text: `gen_{task_id}.txt`
- Task IDs follow format: `{dimension}_{reasoning_type}_{number}`

### 4. Run Evaluation

```bash
# Evaluate all available results
python evaluate_vortex.py --output_dir results

# Filter by reasoning type
python evaluate_vortex.py --output_dir results --reasoning_type temporal

# Filter by dimension
python evaluate_vortex.py --output_dir results --dimension science

# Custom worker count
python evaluate_vortex.py --output_dir results --workers 5
```

### 5. View Results

```bash
# Generate summary report
python summarize.py --results_file results/vortex_metrics.jsonl --output_dir results

# View detailed results
cat results/vortex_summary.json
```

## Evaluation Metrics

The system evaluates 5 core metrics:

| Metric | Code | Description | Requires Text |
|--------|------|-------------|---------------|
| **Reasoning Process** | RP | Quality of written reasoning steps | ✅ Yes |
| **Reasoning Visual** | RV | Visual result matches target description | ❌ No |
| **Reasoning Alignment** | RA | Consistency between text and visual result | ✅ Yes |
| **Visual Consistency** | VC | Non-target elements remain unchanged | ❌ No |
| **Image Quality** | IQ | Technical quality of generated image | ❌ No |

## Missing Text Files Handling

**The system gracefully handles missing reasoning text files:**

- ✅ **3 visual metrics** (RV, VC, IQ) work normally without text
- ⚠️ **2 text-dependent metrics** (RP, RA) use fallback: "No think output available"
- 📊 **Scoring remains fair**: Missing text results in appropriate score penalties for text-dependent metrics

**Example without text files:**
```
your_gen_dir/
├── gen_science_temporal_1.png    # Only image, no .txt
├── gen_science_causal_3.png      # Only image, no .txt
└── ...
```
→ Still evaluates RV, VC, IQ normally; RP and RA get low scores due to missing reasoning.

## Reasoning Types

- **Temporal**: Changes over time (growth, aging, weather transitions)
- **Spatial**: Geometric transformations (rotation, perspective, positioning)  
- **Quantitative**: Numerical changes (counting, scaling, proportions)
- **Causal**: Cause-effect relationships (interventions, reactions)
- **Synthetic**: Creative additions/modifications (style transfer, object addition)

## Dimensions

- **Science**: Physics, chemistry, biology principles
- **Humanity**: Cultural, historical, social contexts
- **Common Sense**: Everyday knowledge and practical understanding
- **Logic**: Mathematical and formal reasoning

## Output Format

**Raw Results:** `results/vortex_metrics.jsonl` - One JSON object per evaluated task

**Summary Report:** `results/vortex_summary.json` - Aggregated statistics by dimension, reasoning type, and overall

**Console Output:** Formatted table with scores (0-100 scale) for easy reading

## Command Line Options

```bash
python evaluate_vortex.py [OPTIONS]

--output_dir DIR              Output directory (default: vortex_results)
--workers N                   Number of parallel workers (default: 10)
--dimension {science,humanity,common_sense,logic}
                             Filter by dimension
--reasoning_type {temporal,spatial,quantitative,causal,synthetic}
                             Filter by reasoning type
--metrics METRIC [METRIC ...]
                             Specific metrics to evaluate
```

## Data Source

The evaluation uses the `cheryyunl/ROVER-Gen` dataset from Hugging Face, which is automatically downloaded. No manual data preparation needed.

## Requirements

- Python 3.7+
- OpenAI API access
- Required packages: `datasets`, `openai`, `PIL`, `tqdm`

## Troubleshooting

**Common Issues:**

1. **"Generated image not found"** → Check `VORTEX_GEN_DIR` path and file naming
2. **API errors** → Verify OpenAI credentials in `config.py`
3. **Low RP/RA scores** → Normal if reasoning text files (.txt) are missing
4. **Dataset loading fails** → Check internet connection for Hugging Face access
