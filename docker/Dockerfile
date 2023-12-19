FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate the environment, and make sure it's activated:
# RUN conda activate env

# RUN echo "conda activate env" >> ~/.bashrc
# SHELL ["/bin/bash", "--login", "-c"]

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]

RUN python -m spacy download en_core_web_sm
RUN pip install git+https://github.com/hypercycle-development/pyhypercycle-aim.git#egg=pyhypercycle_aim

# The code to run when container is started:
COPY . .
ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "env", "docker/entrypoint.sh" ]