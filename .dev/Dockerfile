FROM pangeo/pangeo-notebook:2022.12.18

COPY environment-dev.yml /tmp/env.yaml
RUN mamba update -y -n notebook -f /tmp/env.yaml && \
    mamba clean --all --yes
