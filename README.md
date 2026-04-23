# OnlineSurvival

Clean reference implementations for the methods requested from
`online_survival.tex`:

- `ACIWithIPCW`
- `ACIWithoutIPCW`
- `AdaFTRL`
- `AdaFTRLV2` / `adaftrl_v2`

The implementations live in `online_survival/algorithms.py`. A dependency-light
synthetic experiment runner lives in `online_survival/experiments.py`.

Run the experiment notebook:

```bash
jupyter notebook notebooks/aci_adaftrl_experiments.ipynb
```

Run the heavy-censoring notebook where IPCW is much more useful:

```bash
jupyter notebook notebooks/ipcw_heavy_censoring_advantage.ipynb
```

Run the ambiguity-neutral AdaFTRL showcase:

```bash
jupyter notebook notebooks/adaftrl_v2_showcase.ipynb
```

Run the long censor-only stress notebook:

```bash
jupyter notebook notebooks/adaftrl_v2_censor_only_stress.ipynb
```

Run the multi-situation stress suite:

```bash
jupyter notebook notebooks/all_methods_stress_suite.ipynb
```
