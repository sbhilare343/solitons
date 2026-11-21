# PatchTST, rolling forecasts, and information leakage

## Plain language

PatchTST turns a recent history into a short sequence of overlapping chunks (“patches”). A
Transformer encoder lets each patch representation use information from the other patches; a
small linear head maps the encoded sequence to the next return. For one return series there is one
channel. The model has many parameters relative to the weak signal in daily returns, so training
fit does not guarantee forecasting value.

At forecast origin t, the input is exactly the previous L observed values. The forecast is made
before r_t is revealed. After r_t becomes observed, it can appear in the next origin's input. The
weights and scaler remain fixed. Z-score statistics are calculated on training data only.

## Mathematics

For lookback L, patch length p and stride s, the number of patches is

`P = floor((L-p)/s) + 1`.

Patch j is `x[j*s : j*s+p]`; a linear map projects each p-vector into d dimensions. Add position
embeddings, apply a Transformer encoder, flatten the P×d array, and predict one scaled target.
With EXP-005, L=96, p=16, s=8, so P=11. The parameter count is 69,441.

The scaler is `z_t=(r_t-mean_train)/sd_train`; the output is returned to percentage-point units
as `rhat_t=sd_train*z_hat_t+mean_train`. Training minimizes mean squared scaled error. Phase 1
selects an epoch using only the last 15% of the training partition; phase 2 initializes a fresh
seeded model and trains on all training windows for that fixed number of epochs.

## Numerical example

Suppose the three newest standardized inputs are `[1, 2, 3]`, patch length 2 and stride 1.
Patches are `[1,2]` and `[2,3]`, hence P=2. If the training scaler mean is 0.04 and SD is 1.10
percentage points, a scaled forecast of 0.20 maps to `0.04 + 1.10×0.20 = 0.26` percentage
points. A window for target t excludes r_t; perturbing r_t must not change the forecast at t.

## Reading EXP-005

The registered model's validation RMSE is 1.2952 percentage points, worse than zero's 1.1618.
That is evidence against this small fixed PatchTST configuration on this validation period. It
does not establish that all Transformer architectures fail. The model warning about PyTorch's
nested-tensor fast path is a runtime optimization warning, not a fit failure.
