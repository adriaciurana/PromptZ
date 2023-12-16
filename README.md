# GA-prompt-LLM
Generate prompts using GA algorithm for a pretrained LLM

# Install
For now we can use conda, I tried to use pipenv with pytorch but works bad :( (Frozen when tries to lock)
```python
# conda create -p "$(pwd)/env" python=3.11
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# pip install transformers
# pip install -r hypercycle_requirements.txt
# pip install pytest-xprocess
# pip install accelerate
# pip install bitsandbytes
# pip install scipy
# pip install sentence_transformers
# pip install qqdm
# pip install tokenizers

conda install --file environment.yaml
```