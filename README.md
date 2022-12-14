# Surnamerator

Surnamerator is a surname generator. It's a testbed for different network architectures inspired by [Andrej Karpathy's zero-to-hero youtube series](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ). 

Live demo at the [surnamerator huggingface space](https://huggingface.co/spaces/jefsnacker/surnamerator)!


## Getting started
To run existing models run `app.ipynb` in a jupyter notebook IDE.
To train a new GPT model,  run `lit_surnames.ipynb`.


## File Structure 
* `app.ipynb` - example code for the huggingface gradio
* `lit_surnames.ipynb` - Most up to date training file for the transformer architecture.
* `surnamerator.py` - Library for network models and utils.
* `first_names.txt`/`surnames.txt` - Dataset.
* `data/` - Dataset creation.
* `prototypes/` - Network architecture prototypes.
* `models/` - Saved weights and configs.


### Dataset
Dataset creation for `surnames.txt` is done in `./data`. `first_names.txt` was downloaded from [makemore](https://github.com/karpathy/makemore/blob/master/names.txt). Example surname data:

```
butterly
gawne
mouch
...
```

The raw data for surnames was found at [fivethirtyeight](https://github.com/fivethirtyeight/data/tree/master/most-common-name), [data.world](https://data.world/crowdflower/transc-names-from-handwriting), and [cencus.gov](https://www.census.gov/topics/population/genealogy/data/2010_surnames.html).
