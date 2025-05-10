# KAN Training Results Summary

## Basic Metrics

| Metric     | Standard Value | Quantized Value |
| ---------- | -------------- | --------------- |
| PSNR       | 30.399204      | 30.361235       |
| Loss       | 0.00253466     | 0.00254279      |
| Distortion | 0.00091218     | 0.00092019      |
| SSIM       | 0.8857845      | 0.8849787       |

## Rate and Bits Per Pixel (BPP) Metrics

| Metric              | Rate Value | BPP Value  |
| ------------------- | ---------- | ---------- |
| Latents             | 637984.9   | 1.6224794  |
| Latents (Quantized) | 638032.0   | 1.6225994  |
| Synthesis           | 9230.976   | 0.02347559 |
| Entropy             | 9448.096   | 0.02402775 |
| Total               | 656711.1   | 1.6701028  |

## Quantization Parameters

| Parameter     | Value  |
| ------------- | ------ |
| Q Step Weight | 0.001  |
| Q Step Bias   | 0.0001 |

## Computational Complexity (MACs per pixel)

| Component                | MACs |
| ------------------------ | ---- |
| Interpolation            | 48   |
| Entropy Model            | 1056 |
| Synthesis                | 666  |
| Total (no interpolation) | 1722 |
| Total                    | 1770 |

## Image Size Information

| Parameter         | Value    |
| ----------------- | -------- |
| Number of Pixels  | 393216   |
| Number of Latents | 524256.0 |

## Training Progress

- **Global Step**: 1
- **Steps per Second**: 0.0002922580942977078
- **Training Status**: Completed
- **Checkpoint**: Saved with id 0

## Training Log Summary

The training completed successfully with these final metrics at global step 1. The model was trained with a decreasing learning rate schedule until reaching below threshold 1.00000e-08. Quantization step search was performed and completed.

After Straight-Through Estimator (STE) was enabled, the best loss achieved was 2.5334032252e-03 with a final learning rate of 8.5071e-09.

## Key Training Events

```
I0506 21:45:10.949826 - Switched to STE
I0506 21:45:10.950051 - Best loss before STE: 2.5372649543e-03
I0506 21:47:03.451979 - Best loss after STE: 2.5334032252e-03
I0506 21:47:06.376407 - Started quantization step search
I0506 21:47:31.728734 - Finished quantization step search
I0506 21:47:31.831001 - Training loop finished
```
