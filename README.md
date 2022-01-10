# TSLM-DISCOURSE-MARKERS

<!-- Not always needed, but a scope helps the user understand in a short sentance like below, why this repo exists -->
## Scope

This repository contains:

 (1) Code to extract discourse markers from wikipedia (TSA).

 (1) Code to extract significant discoßurse markers from predictions over a sample

## Usage

**Evaluation code**: 

<ins>Installation</ins>

Using pip:
```
pip install git+ssh://git@github.com/IBM/tslm-discourse-markers.git#egg=tslm-discourse-markers
```

Alternatively, you can first clone the code, and install the requirements: 

```commandline
1. git clone git@github.com:IBM/tslm-discousrse-markers.git
2. cd tslm-discourse-markers
3. pip install -r requirements.txt
```
You also need to download fasttext model:
curl https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -o ~/Downloads/lid.176.bin
and spacy english model:
python -m spacy download en_core_web_sm

<ins>Running</ins>

## Citing tslm-discourse-markers

If you are using tslm-discourse-markers in a publication, please cite the following paper:

Liat Ein-Dor, Ilya Shnayderman, Artem Spector, Lena Dankin,Ranit Aharonov and Noam Slonim 2022
[Fortunately, Discourse Markers Can Enhance Language Models for Sentiment Analysis](https://arxiv.org/abs/2201.02026). AAAI-2022.  

## Loading dataset
import datasets

directory = 'dataset/WIKI_ENGLISH'
datasets.load_dataset('csv', data_files={folder: [f'{directory}/{folder}/{folder}_*.csv.gz'] for folder in ['train', 'dev','test']})


## Contributing

This project welcomes external contributions, if you would like to contribute please see further instructions [here](CONTRIBUTING.md)

Pull requests are very welcome! Make sure your patches are well tested.
Ideally create a topic branch for every separate change you make. For
example:

1. Fork the repo
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Added some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create new Pull Request

## Changelog

<!-- A Changelog allows you to track major changes and things that happen, https://github.com/github-changelog-generator/github-changelog-generator can help automate the process -->
Major changes are documented [here](CHANGELOG.md).

<!-- The following are OPTIONAL, but strongly suggested to have in your repository. 
* [dco.yml](.github/dco.yml) - This enables DCO bot for you, please take a look https://github.com/probot/dco for more details.
* [travis.yml](.travis.yml) - This is a example `.travis.yml`, please take a look https://docs.travis-ci.com/user/tutorial/ for more details.
-->

<!-- A notes section is useful for anything that isn't covered in the Usage or Scope. Like what we have below. -->
## Notes

<!--
**NOTE: This repository has been configured with the [DCO bot](https://github.com/probot/dco).
When you set up a new repository that uses the Apache license, you should
use the DCO to manage contributions. The DCO bot will help enforce that.
Please contact one of the IBM GH Org stewards.**
-->

If you have any questions or issues you can create a new [issue here][issues].

## License

This code is distributed under Apache License 2.0. If you would like to see the detailed LICENSE click [here](LICENSE).

## Authors

The YASO dataset was collected by Liat Ein-Dor, Ilya Shnayderman, Artem Spector, Lena Dankin, Ranit Aharonov and Noam Slonim.

The code was written by [Ilya Shnayderman](https://github.com/ilyashnil).

[issues]: https://github.com/IBM/tslm-discouse-markers/issues/new
