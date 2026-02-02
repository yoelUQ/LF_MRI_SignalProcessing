# MRI Noise Reduction Baselines (MATLAB)

This repository contains MATLAB implementations of multiple baseline models for
coil-based noise estimation and suppression in low-field MRI experiments.

The methods explore linear and neural-network-based mappings from multi-coil,
multi-parameter inputs to a target coil signal across frequency bins.

---

## Data Format

All models assume input data stored in `.mat` files with the following structure:

- `src_data` : `(N × 51 × 15)`
  - `N`  = number of samples
  - `51` = frequency bins
  - `15` = 3 coils × 5 parameters per coil
- `tgt_data` : `(N × 51)`
  - Target signal for the reference coil
- `xfreq` : `(51 × 1)`
  - Frequency axis (kHz)

Example file name:
CNN_input_5ms_abs_080kHz.mat


---

## Implemented Models

### 1. Linear Baseline
**File:** `src/run_linear_baseline.m`

- Multivariate linear regression per frequency bin
- Uses all coils and parameters to predict the target coil
- Serves as the main interpretability and performance baseline

**Purpose:**  
Establish how much noise can be explained by linear inter-coil correlations.

---

### 2. Dense Neural Network Baseline
**File:** `src/run_dense_baseline.m`

- Fully connected feed-forward neural network
- Operates on flattened `(51 × 15)` inputs
- Predicts the full `(51)` frequency response of the target coil

**Purpose:**  
Test whether mild non-linearities improve over the linear model.

---

### 3. 1D CNN Baseline
**File:** `src/run_cnn1d_baseline.m`

- 1D convolutional neural network along the frequency dimension
- Treats parameters/coils as channels
- Learns local spectral correlations

**Purpose:**  
Capture frequency-local structure not accessible to dense models.

---

## Running Experiments

### Run all baselines
**File:** `scripts/run_all_baselines.m`

Runs all implemented models sequentially on a selected dataset and
stores performance metrics (MSE, correlation, SNR gain).

```matlab
run_all_baselines
