<p align="center">
  <img src="/Llama_Repo.jpeg" width="400"/>
</p>

<p align="center">
        🤗 <a href="https://huggingface.co/meta-Llama"> Models on Hugging Face</a>&nbsp | <a href="https://ai.meta.com/blog/"> Blog</a>&nbsp |  <a href="https://llama.meta.com/">Website</a>&nbsp | <a href="https://llama.meta.com/get-started/">Get Started</a>&nbsp
<br>

---

## Fork Announcement and Key Enhancements

This repository is a public fork of the official [meta llama-models](https://github.com/meta-llama/llama-models) repository. In this fork, we implement a Chain-of-Thought (CoT) decoding approach for reasoning tasks as presented in the paper:

**"Chain-of-Thought Reasoning Without Prompting"**  
*Xuezhi Wang, Denny Zhou – DeepMind (2024)*

### What It Does

The new CoT decoding feature allows the model to internally generate multiple reasoning paths and pick the one that produces the **answer tokens** (i.e. only those tokens indicating the answer, such as a number or an option for a math problem) with the highest confidence. The confidence score is defined as the difference in probability between the best predicted token and the second best at each decoding step. When multiple answer tokens are generated, their average confidence score is used. This enhancement is designed to improve multi-step reasoning performance, particularly for tasks that require complex, step-by-step answers.

### Performance Results

- **Original Approach:** On a randomly sampled GSM8K benchmark, Llama 3.1 8B Instruct achieves 83.0% accuracy. 
- **With CoT Decoding:** Llama 3.1 8B Instruct + CoT decoding achieves 90.0% accuracy on the same test set.  
  *Note:* This improvement comes with the overhead of decoding multiple paths.
  Also note that Llama 3.1 8B Instruct might have been pre-trained on GSM8K, so the actual improvement may be different on an unseen reasoning dataset. 

### Example Usage

To enable CoT reasoning, simply pass the `--cot_decoding true` flag along with your desired `--cot_top_k` value when running the model. 

Run script 
```bash
python models/scripts/example_cot_decode.py $CHECKPOINT_DIR --cot_decoding true --cot_top_k 10
```

The default setting assumes that the answer appears on a new line enclosed within answer tags. For example:

```
Mary worked for 4 hours. She was paid $10 per hour. So the total payment is $40
<answer>40</answer>
```
The answer tags can be modified using the `answer_start_pattern` and `answer_end_pattern` options in the `generate()` function.


**Note:**  
To do : The script currently prints <|eot_id|><|finetune_right_pad_id|>.. in the response. Please ignore or filter these out as work around. 

Dependency - datasets module.
```
pip install datasets
```

This prototype has only been tested on a single A100 GPU and has not been optimized for runtime or memory usage. Multi-GPU parallelism and tensor parallelism have not been validated. Only tested on Llama 3.1 8B Instruct which does not support vision. The provided script demonstrates how to use the model for GSM8K evaluation. This prototype is intended solely for research purposes and benchmarking. 



# Llama Models

Llama is an accessible, open large language model (LLM) designed for developers, researchers, and businesses to build, experiment, and responsibly scale their generative AI ideas. Part of a foundational system, it serves as a bedrock for innovation in the global community. A few key aspects:
1. **Open access**: Easy accessibility to cutting-edge large language models, fostering collaboration and advancements among developers, researchers, and organizations
2. **Broad ecosystem**: Llama models have been downloaded hundreds of millions of times, there are thousands of community projects built on Llama and platform support is broad from cloud providers to startups - the world is building with Llama!
3. **Trust & safety**: Llama models are part of a comprehensive approach to trust and safety, releasing models and tools that are designed to enable community collaboration and encourage the standardization of the development and usage of trust and safety tools for generative AI

Our mission is to empower individuals and industry through this opportunity while fostering an environment of discovery and ethical AI advancements. The model weights are licensed for researchers and commercial entities, upholding the principles of openness.

## Llama Models

[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-models)](https://pypi.org/project/llama-models/)
[![Discord](https://img.shields.io/discord/1257833999603335178)](https://discord.gg/TZAAYNVtrU)

|  **Model** | **Launch date** | **Model sizes** | **Context Length** | **Tokenizer** | **Acceptable use policy**  |  **License** | **Model Card** |
| :----: | :----: | :----: | :----:|:----:|:----:|:----:|:----:|
| Llama 2 | 7/18/2023 | 7B, 13B, 70B | 4K | Sentencepiece | [Use Policy](models/llama2/USE_POLICY.md) | [License](models/llama2/LICENSE) | [Model Card](models/llama2/MODEL_CARD.md) |
| Llama 3 | 4/18/2024 | 8B, 70B | 8K | TikToken-based | [Use Policy](models/llama3/USE_POLICY.md) | [License](models/llama3/LICENSE) | [Model Card](models/llama3/MODEL_CARD.md) |
| Llama 3.1 | 7/23/2024 | 8B, 70B, 405B | 128K | TikToken-based | [Use Policy](models/llama3_1/USE_POLICY.md) | [License](models/llama3_1/LICENSE) | [Model Card](models/llama3_1/MODEL_CARD.md) |
| Llama 3.2 | 9/25/2024 | 1B, 3B | 128K | TikToken-based | [Use Policy](models/llama3_2/USE_POLICY.md) | [License](models/llama3_2/LICENSE) | [Model Card](models/llama3_2/MODEL_CARD.md) |
| Llama 3.2-Vision | 9/25/2024 | 11B, 90B | 128K | TikToken-based | [Use Policy](models/llama3_2/USE_POLICY.md) | [License](models/llama3_2/LICENSE) | [Model Card](models/llama3_2/MODEL_CARD_VISION.md) |

## Download

To download the model weights and tokenizer:

1. Visit the [Meta Llama website](https://llama.meta.com/llama-downloads/).
2. Read and accept the license.
3. Once your request is approved you will receive a signed URL via email.
4. Install the [Llama CLI](https://github.com/meta-llama/llama-stack): `pip install llama-stack`. (**<-- Start Here if you have received an email already.**)
5. Run `llama model list` to show the latest available models and determine the model ID you wish to download. **NOTE**:
If you want older versions of models, run `llama model list --show-all` to show all the available Llama models.

6. Run: `llama download --source meta --model-id CHOSEN_MODEL_ID`
7. Pass the URL provided when prompted to start the download.

Remember that the links expire after 24 hours and a certain amount of downloads. You can always re-request a link if you start seeing errors such as `403: Forbidden`.

## Running the models

You need to `pip install llama_models[torch]` to run the models. After installing the dependencies, you can run the example scripts (within `llama_models/scripts/` sub-directory) as follows:
```bash
#!/bin/bash

CHECKPOINT_DIR=~/.llama/checkpoints/Meta-Llama3.1-8B-Instruct
PYTHONPATH=$(git rev-parse --show-toplevel) torchrun llama_models/scripts/example_chat_completion.py $CHECKPOINT_DIR
```

The above script should be used with an Instruct (Chat) model. For a Base model, update the `CHECKPOINT_DIR` path and use the script `llama_models/scripts/example_text_completion.py`. Note that you can use these scripts with both Llama3 and Llama3.1 series of models.

For running larger models with tensor parallelism, you should modify as:
```bash
#!/bin/bash

NGPUS=8
PYTHONPATH=$(git rev-parse --show-toplevel) torchrun \
  --nproc_per_node=$NGPUS \
  llama_models/scripts/example_chat_completion.py $CHECKPOINT_DIR \
  --model_parallel_size $NGPUS
```

For more flexibility in running inference (including running FP8 inference), please see the [`Llama Stack`](https://github.com/meta-llama/llama-stack) repository.


## Access to Hugging Face

We also provide downloads on [Hugging Face](https://huggingface.co/meta-llama), in both transformers and native `llama3` formats. To download the weights from Hugging Face, please follow these steps:

- Visit one of the repos, for example [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct).
- Read and accept the license. Once your request is approved, you'll be granted access to all Llama 3.1 models as well as previous versions. Note that requests used to take up to one hour to get processed.
- To download the original native weights to use with this repo, click on the "Files and versions" tab and download the contents of the `original` folder. You can also download them from the command line if you `pip install huggingface-hub`:

```bash
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --include "original/*" --local-dir meta-llama/Meta-Llama-3.1-8B-Instruct
```

**NOTE** The original native weights of meta-llama/Meta-Llama-3.1-405B would not be available through this HugginFace repo.


- To use with transformers, the following [pipeline](https://huggingface.co/docs/transformers/en/main_classes/pipelines) snippet will download and cache the weights:

  ```python
  import transformers
  import torch

  model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

  pipeline = transformers.pipeline(
      "text-generation",
      model="meta-llama/Meta-Llama-3.1-8B-Instruct",
      model_kwargs={"torch_dtype": torch.bfloat16},
      device="cuda",
  )
  ```

## Installations

You can install this repository as a [package](https://pypi.org/project/llama-models/) by just doing `pip install llama-models`

## Responsible Use

Llama models are a new technology that carries potential risks with use. Testing conducted to date has not — and could not — cover all scenarios.
To help developers address these risks, we have created the [Responsible Use Guide](https://ai.meta.com/static-resource/responsible-use-guide/).

## Issues

Please report any software “bug” or other problems with the models through one of the following means:
- Reporting issues with the model: [https://github.com/meta-llama/llama-models/issues](https://github.com/meta-llama/llama-models/issues)
- Reporting risky content generated by the model: [developers.facebook.com/llama_output_feedback](http://developers.facebook.com/llama_output_feedback)
- Reporting bugs and security concerns: [facebook.com/whitehat/info](http://facebook.com/whitehat/info)


## Questions

For common questions, the FAQ can be found [here](https://llama.meta.com/faq), which will be updated over time as new questions arise.
