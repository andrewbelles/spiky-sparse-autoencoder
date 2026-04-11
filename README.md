# Spiky-Autoencoder with Sparsity 

Key to compressed sensing is the sparse assumption. The idea that human intelligence is compression inspired me to make this proof of concept coupling the Spiky Neural Network architecture, modeled after the brain, with a Sparse Auto Encoder to learn a compressed representation of music samples from the FMA dataset.

### Data Processing 

The data used to train the SNN + SAE model comes from the *Free Music Archive (FMA)* dataset. Currently work reflected by this repository was done using the `fma_small` dataset. The dataset `fma_small` spans 8 genres, with 1,000 tracks for each genre. The genres: 
- Electronic 
- Experimental 
- Folk
- Hip-Hop 
- Instrumental 
- International
- Pop 
- Rock 

The pipeline converts the `.mp3` files into a mel-spectrogram using `torchaudio`. I use $64$ mel bins and downsample audio to $22.05 kHz$ for VRAM efficiency (Running natively on GTX 4060, 8 GiB VRAM) and $22.05 kHz$ is a good candidate for satisfying the Nyquist-Shannon sample bound. Likewise we log-scale the mel-spectrogram then apply a min-max scale per track so that all pixel values are bounded in $[0,1]$.

The mel-spectrograms are saved to pytorch tensors (`.pt` format) to `preprocess/data/fma_{size}_mel/`. These are the files used by the autoencoder. 

### Preprocess Usage  

Configs for each stage of pipeline are located in `configs/`. 

Download the FMA metadata and extract `tracks.csv` using: 

```bash
bash -v preprocess/metadata.sh 
```

Download the `fma_small` dataset and extract using: 

```bash
bash -v preprocess/small.sh 
```

Then convert to mel-spectrogram using: 

```bash
python -m preprocess.mel -d preprocess/data/fma_small 
```

Pass the flag `--sample-images` to generate a single image for each genre. Optionally a different config can be passed via `-c configs/mel.yaml`.

### Representation Usage 

Learned representations can be trained using their configuration files. Example configs are outlined in `configs/`. All AE models use a ResNet-style CNN architecture for hidden layers with special care for the Spiky AE implementation (Since it structurally differs from an ANN). As an example, to train and generate representations using the training samples (and extrapolate to validation and test samples): 

```bash
python -m manifold.standard -d preprocess/data/fma_small_mel -c configs/standard.yaml
```

With the other AE models following the same pattern. I have defined the following model types:
- Standard 
- Spiky with KL divergence (Sparse)
- Sparse 
- Variational 

There is a UMAP visualization tool in `manifold/` that shows a 2D projection of the learned representation with coloring by genre to visually confirm linear separability. It can be called via: 

```bash
python -m manifold.visualizations -c configs/visualizations.yaml
```

Where the config just specifies which AE methods to generate UMAP projections for. 

### Citation 

This dataset was made possible by the incredible work of Defferrard et al. If you are doing downstream research using this codebase, please ensure you also credit the original FMA authors:

> Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson.  
> **"FMA: A Dataset for Music Analysis"** > *18th International Society for Music Information Retrieval Conference (ISMIR), 2017.* > [Official FMA GitHub Repository](https://github.com/mdeff/fma)
