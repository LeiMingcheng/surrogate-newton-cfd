# syntax=docker/dockerfile:1.7

ARG UBUNTU_IMAGE=ubuntu:24.04

FROM ${UBUNTU_IMAGE} AS miniforge

ARG MINIFORGE_VERSION=26.3.2-2
ARG MINIFORGE_SHA256=42260ffe3830fb953d5eee1bbb32229ff06aa7c3833c1ed7a9a0420a95685d94

RUN apt-get update \
    && apt-get install --yes --no-install-recommends bzip2 ca-certificates curl \
    && curl --fail --location --show-error \
        --output /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
    && printf '%s  %s\n' "${MINIFORGE_SHA256}" /tmp/miniforge.sh | sha256sum --check - \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm -f /tmp/miniforge.sh \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_ENV=/opt/conda/envs/surrogate-newton
ENV PATH=${CONDA_ENV}/bin:/opt/conda/bin:${PATH}

FROM miniforge AS runtime-environment

COPY deployment/container/environment.runtime.yml /tmp/environment.runtime.yml
RUN conda config --system --set channel_priority flexible \
    && conda env create --prefix ${CONDA_ENV} --file /tmp/environment.runtime.yml \
    && conda clean --all --yes

FROM runtime-environment AS builder

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY deployment/container/environment.build.yml /tmp/environment.build.yml
RUN conda env update --prefix ${CONDA_ENV} --file /tmp/environment.build.yml \
    && conda clean --all --yes

WORKDIR /opt/build-context
COPY solver-stack.lock.yaml ./solver-stack.lock.yaml
COPY deployment/container/config ./deployment/container/config
COPY scripts/fetch_solver_stack.sh scripts/build_solver_stack.sh scripts/verify_solver_stack.sh ./scripts/

ENV SURROGATE_NEWTON_ENV_PREFIX=${CONDA_ENV}
ENV SURROGATE_NEWTON_CGNS_PREFIX=/opt/cgns
ENV SURROGATE_NEWTON_SOLVER_LOCK=/opt/build-context/solver-stack.lock.yaml
ENV LD_LIBRARY_PATH=/opt/cgns/lib:${CONDA_ENV}/lib

RUN scripts/fetch_solver_stack.sh /opt/solver-stack \
    && scripts/build_solver_stack.sh /opt/solver-stack \
    && scripts/verify_solver_stack.sh /opt/solver-stack

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN mkdir -p /opt/solver-export/pyhyp /opt/solver-export/adflow \
    && git -C /opt/solver-stack/pyhyp archive HEAD \
        LICENSE.md README.md SURROGATE_NEWTON_CHANGES.md pyhyp \
        | tar -x -C /opt/solver-export/pyhyp \
    && git -C /opt/solver-stack/adflow archive HEAD \
        LICENSE.md README.md SURROGATE_NEWTON_CHANGES.md adflow src/f2py \
        | tar -x -C /opt/solver-export/adflow \
    && cp /opt/solver-stack/pyhyp/pyhyp/hyp.so /opt/solver-export/pyhyp/pyhyp/hyp.so \
    && cp /opt/solver-stack/adflow/adflow/libadflow.so /opt/solver-export/adflow/adflow/libadflow.so \
    && git -C /opt/solver-stack/pyhyp rev-parse HEAD > /opt/solver-export/pyhyp/.surrogate-newton-revision \
    && git -C /opt/solver-stack/adflow rev-parse HEAD > /opt/solver-export/adflow/.surrogate-newton-revision

COPY . /opt/surrogate-newton-source
RUN ${CONDA_ENV}/bin/python -m pip wheel \
    --no-deps --no-build-isolation \
    --wheel-dir /opt/wheels \
    /opt/surrogate-newton-source

FROM ${UBUNTU_IMAGE} AS runtime

ARG VCS_REF=unknown
ARG IMAGE_VERSION=0.1.0-rc1
ARG SOURCE_URL=https://github.com/LeiMingcheng/surrogate-newton-cfd

LABEL org.opencontainers.image.title="Surrogate-Newton CFD runtime" \
      org.opencontainers.image.description="RAE2822 surrogate inference and ADflow Newton-Krylov correction runtime" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.licenses="BSD-3-Clause AND Apache-2.0 AND LGPL-2.1-only"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates curl libgomp1 \
    && useradd --create-home --home-dir /home/surrogate --uid 10001 surrogate \
    && mkdir -p /models /runtime /opt/surrogate-newton \
    && chown surrogate:surrogate /runtime \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_ENV=/opt/conda/envs/surrogate-newton
ENV PATH=${CONDA_ENV}/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/cgns/lib:${CONDA_ENV}/lib
ENV PYTHONNOUSERSITE=1
ENV PYTHONPATH=/opt/solver-stack/pyhyp:/opt/solver-stack/adflow:/opt/surrogate-newton/src
ENV SURROGATE_NEWTON_ADFLOW_ROOT=/opt/solver-stack/adflow
ENV SURROGATE_NEWTON_PYTHON=${CONDA_ENV}/bin/python
ENV SURROGATE_NEWTON_MODEL_DIR=/models
ENV SURROGATE_NEWTON_RUNTIME_DIR=/runtime
ENV OMP_NUM_THREADS=1

COPY --from=runtime-environment /opt/conda/envs/surrogate-newton ${CONDA_ENV}
COPY --from=builder /opt/cgns /opt/cgns
COPY --from=builder /opt/solver-export /opt/solver-stack
COPY --from=builder /opt/wheels /tmp/wheels
COPY . /opt/surrogate-newton/src

RUN ${CONDA_ENV}/bin/python -m pip install --no-deps /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

WORKDIR /opt/surrogate-newton/src
USER surrogate
ENTRYPOINT ["/opt/surrogate-newton/src/deployment/container/entrypoint.sh"]
CMD ["source-smoke"]
