# Use the official devcontainers Conda image
FROM mcr.microsoft.com/devcontainers/miniconda:0-3

# Install additional packages (optional)
RUN conda install -n base -c conda-forge \
    jupyterlab ipykernel pytest numpy pandas matplotlib

# Set environment variables (optional)
# ENV PYTHONUNBUFFERED=1

# Copy default .env and settings.json if not present in workspace
RUN cp -n /workspaces/StructureRelations/.devcontainer/DevCont.env /workspaces/StructureRelations/.env || true
RUN mkdir -p /workspaces/StructureRelations/.vscode && \
    cp -n /workspaces/StructureRelations/.devcontainer/settings.json /workspaces/StructureRelations/.vscode/settings.json || true
# Old and unused commands
# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
# COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
# RUN /opt/conda/bin/conda config --append channels conda-forge
# RUN /opt/conda/bin/conda init
# Install conda environment from the provided YAML file
# RUN conda env create --name StructureRelations --file="/workspaces/StructureRelations/.devcontainer/StructureRelations.yml"
# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
