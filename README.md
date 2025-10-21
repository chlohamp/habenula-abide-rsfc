# habenula-abide-rsfc
let's do this habenula thing again .. but better ... like with 1500 hand drawn ROIs

### Analysis
We performed a seed-based resting state functional connectiivty analysis in the ABIDE dataset to:
1. Regions of atypical connectivity in individuals diagnosed with ASD compared to neurotypical (NT) controls 
2. Regions that exhibit an altered developmental functional connectivity trajectory (age-variability)
3. Phenotypic variability in connectivity between ASD and NT


### Visualizations
#### Must use:

python 3.8, 3.9, 3.10, and 3.11
nilearn 0.10.2
netneurotools 0.2.4

(I personally use Python 3.9.6)

#### To install the necessary packages: 

Install in this order:

1. First install gradec:
   ```
   pip install git+https://github.com/JulioAPeraza/gradec.git
   ```

2. Then install seaborn:
   ```
   pip install seaborn
   ```

3. Install remaining packages:
   ```
   pip install numpy pandas matplotlib neuromaps nibabel nilearn==0.10.2 surfplot scipy
   ```

4. **Important**: Ensure netneurotools is version 0.2.4:
   ```
   pip install netneurotools==0.2.4
   ```
