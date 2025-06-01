FROM mcr.microsoft.com/devcontainers/anaconda:1-3

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN /opt/conda/bin/conda config --append channels conda-forge
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp
RUN /opt/conda/bin/conda init
# Copy default .env and settings.json if not present in workspace
RUN cp -n /workspaces/StructureRelations/.devcontainer/DevCont.env /workspaces/StructureRelations/.env || true
RUN mkdir -p /workspaces/StructureRelations/.vscode && \
    cp -n /workspaces/StructureRelations/.devcontainer/settings.json /workspaces/StructureRelations/.vscode/settings.json || true
# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
