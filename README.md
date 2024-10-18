# v_PVMismatch
 Vectorized version of SunPower's PVMismatch using dynamic programming. PVMismatch is an explicit IV & PV curve trace calculator for PV system circuits.

 ![image](https://github.com/user-attachments/assets/35ae7565-d3ab-4123-9100-d77006e95215)


## Install & Setup
In a fresh Python virtual environment, simply run:

```
pip install v_pvmismatch
```

### Install using .whl file (Recommended)
Download files from Github and run the latest version of the .whl file using the command below from the parent folder.

```
pip install .\dist\v_pvmismatch-<version>-py3-none-any.whl
```

## Requirements

v_PVMismatch requires Matplotlib, Future, NumPy, and SciPy.

## Other Projects that use PVMismatch

[PVShadeSim](https://github.com/Maxeon-RnD/PVShadeSim) includes a physical model for the PV system and utilizes v_PVMismatch for the electrical modeling. It allows for modeling physical representations of shading on a PV module. Various shade scenarios can be generated with an in-buit database. There are also structured databases for the PV cell and module classes in PVMismatch.

## Current Maintainer at Maxeon Solar

@k-bsub
