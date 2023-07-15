# THExtended

### **T**ransformer-based **H**ighlights **E**xtraction Extended to the news domain articles.

THExtended is a software developed by a team of four students from Politecnico di Torino: Alessio Paone, Flavio Spuri, Luca Zilli, and Luca Agnese. The project builds upon the work of Moreno La Quatra available at https://github.com/MorenoLaQuatra/THExt originally developed for the extraction of highlights in the scientific paper domain. It focuses on extracting highlights from textual documents using neural networks based on Transformers architecture.

The purpose of this project is not necessarily to improve the original work performances, given the limited hardware resources available. Instead, it serves as an academic project developed for a university examination. The team aimed to extend the functionality of the original work by introducing modifications and additional features.

## Extensions Description

The team made two extensions to the original project:

1. **News domain adaptation**: we adapted the THExt framework to the news domain. To do so, we also needed a new way to define a meaningful context, as we lose the intrinsic structure present in the scientific paper.

2. **Combining syntactic and semantic scores**: we analyzed the trade-ed  off between using a semantic score and a similarity score by introducing a novel modified scoring mechanism.

## Installation

Run the following to install the dependancies needed

```python
pip install rouge
pip install transformers datasets accelerate nvidia-ml-py3 sentencepiece evaluate
pip install bert_score
pip install spacy
python -m spacy download en_core_web_lg
pip install -U sentence-transformers
```

## Usage

### Pretrained models on 🤗 Hub:

- Best Model: [Alpha = 0.5](https://huggingface.co/Raffix/THExtended_alpha_05) `Raffix/THExtended_alpha_05`
- [Alpha = 1.0](https://huggingface.co/Raffix/THExtended_alpha_1) `Raffix/THExtended_alpha_1`

### Using pretrained models
Demo available at: 

## Credits

The THExtended project was carried out by the following team members:

- Alessio Paone
- Flavio Spuri
- Luca Zilli
- Luca Agnese

They actively contributed to all the phases of the project, dedicating their efforts and expertise to realize this extended version of the software. The project is developed within the scope of an academic examination, and the team acknowledges the original work of Moreno La Quatra as the foundation for their project.

For more information about the original project, please refer to the [https://github.com/MorenoLaQuatra/THExt](THExt).
