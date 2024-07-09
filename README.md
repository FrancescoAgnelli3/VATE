<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/6295/6295417.png" width="100" />
</p>
<p align="center">
    <h1 align="center">VATE</h1>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/FrancescoAgnelli3/VATE?style=flat&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/FrancescoAgnelli3/VATE?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/FrancescoAgnelli3/VATE?style=flat&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/FrancescoAgnelli3/VATE?style=flat&color=0080ff" alt="repo-language-count">
<p>
<hr>

##  Quick Links

> - [ Overview](#-overview)
> - [ Repository Structure](#-repository-structure)
> - [ Getting Started](#-getting-started)
>   - [ Installation](#-installation)
>   - [ Downloading VATE](#-downloading-VATE)
>   - [ Contrastive model](#-contrastive-model)
> - [ Contributing](#-contributing)
> - [ License](#-license)

---

##  Overview

the Video-Audio-Text for affective Evaluation dataset. VATE collects a wide variety of multimodal data exhibiting a multitude of spontaneous human affective states. It contains 21,871 raw videos together with voice recordings and text transcriptions from numerous emotion evoking interviews. VATE is specifically designed for contrastive self-supervised representation learning of human affective states; it prioritises quantity and quality of data over human labelling of emotions, which constitutes a highly subjective, often inconsistent and controversial aspect of modern affective computing. To highlight the usefulness of our proposal, we release a multimodal encoder employing a contrastive video-language-audio pre-training procedure carried out on the VATE dataset. Experimental results show that such model exhibits sensibly better few-shot generalisation abilities when compared to fully supervised baselines on different downstream tasks.

An overview of the data can be found at

```sh
VATE/output/VATE/metadata.json
```

---

##  Repository Structure

```sh
└── VATE/
    ├── VATE.py
    ├── README.md
    ├── audio.py
    ├── contrastive_model.py
    ├── dataset.py
    ├── dataset_utils.py
    ├── feature_extraction
    │   ├── VATE
    │   ├── collect_yb.py
    │   ├── couples.txt
    │   ├── cut_video.py
    │   ├── input.txt
    │   ├── main.py
    │   └── write_video.py
    ├── main.py
    ├── media.py
    ├── output
    │   └── VATE
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

1. Clone the VATE repository:

```sh
git clone https://github.com/FrancescoAgnelli3/VATE
```

2. Change to the project directory:

```sh
cd VATE
```

3. Install the dependencies:

```sh
pip install -r requirements.txt
```

###  Downloading VATE

Use the following command to download the VATE dataset:

1. Change to the project directory:

```sh
cd feature_extraction
```

2. Download the dataset:

```sh
python main.py
```

The dataset will be downloaded in the folder:

```sh
VATE/feature_extraction/VATE
```

If you want to add other YouTube playlists to the dataset, you can add them to the python file and run: 

```sh
python collect_yb.py
```

And then again:

```sh
python main.py
```

###  Contrastive model

1. To train the contrastive model on the dataset, change to the project directory:

```sh
cd ..
```

2. Train the model:

```sh
python main.py
```

3. The model will be saved in (or it can be directly downloaded, already pre-trained, from) the folder:

```sh
VATE/output/VATE/best_model_contrastive.pt
```

## Contributing

To contribute to the project, please follow these guidelines:

1. Fork the repository and clone it to your local machine.

2. Create a new branch for your feature or bug fix.

3. Make your changes and commit them with descriptive commit messages.

4. Push your branch to your forked repository.

5. Submit a pull request to the main repository.

<!-- 
##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Submit Pull Requests](https://github.com/FrancescoAgnelli3/VATE/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/FrancescoAgnelli3/VATE/discussions)**: Share your insights, provide feedback, or ask questions.
- **[Report Issues](https://github.com/FrancescoAgnelli3/VATE/issues)**: Submit bugs found or log feature requests for VATE.

<details closed>
    <summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your GitHub account.
2. **Clone Locally**: Clone the forked repository to your local machine using a Git client.
   ```sh
   git clone https://github.com/FrancescoAgnelli3/VATE
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

--- -->

##  License

This project is protected under the [MIT LICENSE](https://choosealicense.com/licenses/mit/) License.

