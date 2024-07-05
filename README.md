<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/6295/6295417.png" width="100" />
</p>
<p align="center">
    <h1 align="center">VATEMOT</h1>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/FrancescoAgnelli3/VATEmot?style=flat&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/FrancescoAgnelli3/VATEmot?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/FrancescoAgnelli3/VATEmot?style=flat&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/FrancescoAgnelli3/VATEmot?style=flat&color=0080ff" alt="repo-language-count">
<p>
<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=flat&logo=tqdm&logoColor=black" alt="tqdm">
	<img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white" alt="TensorFlow">
	<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="scikitlearn">
	<img src="https://img.shields.io/badge/Keras-D00000.svg?style=flat&logo=Keras&logoColor=white" alt="Keras">
	<img src="https://img.shields.io/badge/Jinja-B41717.svg?style=flat&logo=Jinja&logoColor=white" alt="Jinja">
	<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=flat&logo=SciPy&logoColor=white" alt="SciPy">
	<img src="https://img.shields.io/badge/Gunicorn-499848.svg?style=flat&logo=Gunicorn&logoColor=white" alt="Gunicorn">
	<img src="https://img.shields.io/badge/Plotly-3F4F75.svg?style=flat&logo=Plotly&logoColor=white" alt="Plotly">
	<br>
	<img src="https://img.shields.io/badge/SymPy-3B5526.svg?style=flat&logo=SymPy&logoColor=white" alt="SymPy">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat&logo=pandas&logoColor=white" alt="pandas">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white" alt="NumPy">
	<img src="https://img.shields.io/badge/Numba-00A3E0.svg?style=flat&logo=Numba&logoColor=white" alt="Numba">
	<img src="https://img.shields.io/badge/JSON-000000.svg?style=flat&logo=JSON&logoColor=white" alt="JSON">
	<img src="https://img.shields.io/badge/Flask-000000.svg?style=flat&logo=Flask&logoColor=white" alt="Flask">
	<img src="https://img.shields.io/badge/Markdown-000000.svg?style=flat&logo=Markdown&logoColor=white" alt="Markdown">
</p>
<hr>

##  Quick Links

> - [ Overview](#-overview)
> - [ Repository Structure](#-repository-structure)
> - [ Getting Started](#-getting-started)
>   - [ Installation](#-installation)
>   - [ Downloading VATEmot](#-downloading-VATEmot)
>   - [ Contrastive model](#-contrastive-model)
> - [ Contributing](#-contributing)
> - [ License](#-license)

---

##  Overview

This is the repository for VATEmot, a rich dataset for affective computing called Video-Audio-Text Emotional (VATEmot). VATEmot is designed to capture a diverse range of data, encompassing various cues to different aspects of human emotions. It also enables researchers and developers to create models that can more effectively understand and respond to human emotions. The dataset includes raw audio-visual data and text transcriptions from 21,871 samples, specifically designed for self-supervised representation learning tasks without manual labels. To demonstrate the validity of the dataset, we trained a backbone model on VATEmot using contrastive learning. Experimental results show that this backbone, when used as a feature extractor for downstream tasks, outperforms coders trained on generic datasets. The weights of the contrastive model are provided in the repository.

An overview of the data can be found at

```sh
VATEmot/output/AVEMOT/metadata.json
```

---

##  Repository Structure

```sh
└── VATEmot/
    ├── AVEMOT.py
    ├── README.md
    ├── audio.py
    ├── contrastive_model.py
    ├── dataset.py
    ├── dataset_utils.py
    ├── feature_extraction
    │   ├── .DS_Store
    │   ├── AVEMOT
    │   │   └── .DS_Store
    │   ├── __pycache__
    │   │   ├── count_faces.cpython-310.pyc
    │   │   ├── cut_video.cpython-310.pyc
    │   │   └── write_video.cpython-310.pyc
    │   ├── collect_yb.py
    │   ├── couples.txt
    │   ├── cut_video.py
    │   ├── input.txt
    │   ├── main.py
    │   └── write_video.py
    ├── main.py
    ├── media.py
    ├── output
    │   ├── .DS_Store
    │   └── AVEMOT
    │       ├── .DS_Store
    │       ├── best_model_contrastive.pt
    │       └── metadata.json
    ├── text.py
    ├── train_test.py
    ├── utils.py
    └── video.py
```

---

##  Getting Started


###  Installation

1. Clone the VATEmot repository:

```sh
git clone https://github.com/FrancescoAgnelli3/VATEmot
```

2. Change to the project directory:

```sh
cd VATEmot
```

3. Install the dependencies:

```sh
pip install -r requirements.txt
```

###  Downloading VATEmot

Use the following command to download the VATEmot dataset:

1. Change the project directory:

```sh
cd feature_extraction
```

2. Download the dataset:

```sh
python main.py
```

The dataset will be downloaded in the folder:

```sh
VATEmot/feature_extraction/AVEMOT
```

If you want to add other YouTube playlist to the dataset, you can add them and run: 

```sh
python collect_yb.py
```

and then again:

```sh
python main.py
```

###  Contrastive model

1. To train the contrastive model on the dataset, change to the project directory:

```sh
cd VATemot
```

2. Train the model:

```sh
python main.py
```

3. The model will be saved (or it can be directly downloaded) in the folder:

```sh
VATEmot/output/AVEMOT/best_model_contrastive.pt
```

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Submit Pull Requests](https://github.com/FrancescoAgnelli3/VATEmot/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/FrancescoAgnelli3/VATEmot/discussions)**: Share your insights, provide feedback, or ask questions.
- **[Report Issues](https://github.com/FrancescoAgnelli3/VATEmot/issues)**: Submit bugs found or log feature requests for Vatemot.

<details closed>
    <summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your GitHub account.
2. **Clone Locally**: Clone the forked repository to your local machine using a Git client.
   ```sh
   git clone https://github.com/FrancescoAgnelli3/VATEmot
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to GitHub**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.

Once your PR is reviewed and approved, it will be merged into the main branch.

</details>

---

##  License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

