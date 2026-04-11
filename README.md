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

### Usage 

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
python preprocess/mel.py -d preprocess/data/fma_small 
```

Pass the flag `--sample-images` to generate a single image for each genre. Optionally a different config can be passed via `-c configs/mel.yaml`.

### Citation 

This dataset was made possible by the incredible work of Defferrard et al. If you are doing downstream research using this codebase, please ensure you also credit the original FMA authors:

> Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson.  
> **"FMA: A Dataset for Music Analysis"** > *18th International Society for Music Information Retrieval Conference (ISMIR), 2017.* > [Official FMA GitHub Repository](https://github.com/mdeff/fma)
