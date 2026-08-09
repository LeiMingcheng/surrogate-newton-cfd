ARG UBUNTU_IMAGE=ubuntu:24.04
ARG SOLVER_SOURCE_MODE=git
ARG MINIFORGE_SOURCE_MODE=download
ARG PYTHON_WHEEL_SOURCE_MODE=download

FROM ${UBUNTU_IMAGE} AS ubuntu-base

ARG UBUNTU_APT_MIRROR=
RUN if [ -n "${UBUNTU_APT_MIRROR}" ]; then \
        mirror=${UBUNTU_APT_MIRROR%/}; \
        sed -i \
            -e "s|http://archive.ubuntu.com/ubuntu/|${mirror}/|g" \
            -e "s|http://security.ubuntu.com/ubuntu/|${mirror}/|g" \
            /etc/apt/sources.list.d/ubuntu.sources; \
    fi

FROM ubuntu-base AS runtime-os

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates curl libegl1 libgl1 libgomp1 libsm6 libx11-6 libxext6 \
        libxkbcommon0 libxrender1 libxt6 \
    && rm -rf /var/lib/apt/lists/*

FROM runtime-os AS miniforge-base

RUN apt-get update \
    && apt-get install --yes --no-install-recommends bzip2 \
    && rm -rf /var/lib/apt/lists/*

ARG MINIFORGE_VERSION=26.3.2-2
ARG MINIFORGE_SHA256=42260ffe3830fb953d5eee1bbb32229ff06aa7c3833c1ed7a9a0420a95685d94
ARG MINIFORGE_FILENAME=Miniforge3-26.3.2-2-Linux-x86_64.sh

FROM miniforge-base AS miniforge-source-download
RUN curl --fail --location --show-error \
        --output /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
    && printf '%s  %s\n' "${MINIFORGE_SHA256}" /tmp/miniforge.sh | sha256sum --check - \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm -f /tmp/miniforge.sh

FROM miniforge-base AS miniforge-source-bundle
RUN --mount=type=bind,from=miniforge-installer,source=.,target=/opt/miniforge-installer,readonly \
    printf '%s  %s\n' \
        "${MINIFORGE_SHA256}" "/opt/miniforge-installer/${MINIFORGE_FILENAME}" \
        | sha256sum --check - \
    && bash "/opt/miniforge-installer/${MINIFORGE_FILENAME}" -b -p /opt/conda

FROM miniforge-source-${MINIFORGE_SOURCE_MODE} AS miniforge

ENV CONDA_ENV=/opt/conda/envs/surrogate-newton
ENV PATH=${CONDA_ENV}/bin:/opt/conda/bin:${PATH}

FROM miniforge AS runtime-environment-core

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_DEFAULT_TIMEOUT=120
ARG PIP_RETRIES=10

COPY deployment/container/environment.runtime.yml /tmp/environment.runtime.yml
RUN conda config --system --set channel_priority flexible \
    && PIP_INDEX_URL="${PIP_INDEX_URL}" \
        PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT}" \
        PIP_RETRIES="${PIP_RETRIES}" \
        conda env create --prefix ${CONDA_ENV} --file /tmp/environment.runtime.yml \
    && conda clean --all --yes

ARG TORCH_VERSION=2.8.0+cu128
ARG TORCH_WHEEL_FILENAME=torch-2.8.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl
ARG TORCH_WHEEL_SHA256=0c96999d15cf1f13dd7c913e0b21a9a355538e6cfc10861a17158320292f5954

FROM runtime-environment-core AS python-wheel-download
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_DEFAULT_TIMEOUT=120
ARG PIP_RETRIES=10
RUN ${CONDA_ENV}/bin/python -m pip install \
        --index-url "${PIP_INDEX_URL}" \
        --timeout "${PIP_DEFAULT_TIMEOUT}" \
        --retries "${PIP_RETRIES}" \
        "torch @ https://download-r2.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=${TORCH_WHEEL_SHA256}"

FROM runtime-environment-core AS python-wheel-bundle
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_DEFAULT_TIMEOUT=120
ARG PIP_RETRIES=10
RUN --mount=type=bind,from=python-wheels-cu128,source=.,target=/opt/python-wheels-cu128,readonly \
    printf '%s  %s\n' \
        "${TORCH_WHEEL_SHA256}" "/opt/python-wheels-cu128/${TORCH_WHEEL_FILENAME}" \
        | sha256sum --check - \
    && ${CONDA_ENV}/bin/python -m pip install \
        --index-url "${PIP_INDEX_URL}" \
        --timeout "${PIP_DEFAULT_TIMEOUT}" \
        --retries "${PIP_RETRIES}" \
        "/opt/python-wheels-cu128/${TORCH_WHEEL_FILENAME}"

FROM python-wheel-${PYTHON_WHEEL_SOURCE_MODE} AS runtime-environment

FROM runtime-environment AS builder-base

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY deployment/container/environment.build.yml /tmp/environment.build.yml
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_DEFAULT_TIMEOUT=120
ARG PIP_RETRIES=10
RUN conda env update --prefix ${CONDA_ENV} --file /tmp/environment.build.yml \
    && ${CONDA_ENV}/bin/python -m pip install \
        --index-url "${PIP_INDEX_URL}" \
        --timeout "${PIP_DEFAULT_TIMEOUT}" \
        --retries "${PIP_RETRIES}" \
        --no-deps cmake==3.31.6 \
    && test "$(cmake --version | head -n 1)" = "cmake version 3.31.6" \
    && conda clean --all --yes

WORKDIR /opt/build-context
COPY solver-stack.lock.yaml ./solver-stack.lock.yaml
COPY deployment/container/config ./deployment/container/config
COPY scripts/fetch_solver_stack.sh scripts/build_solver_stack.sh scripts/verify_solver_stack.sh ./scripts/

ENV SURROGATE_NEWTON_ENV_PREFIX=${CONDA_ENV}
ENV SURROGATE_NEWTON_CGNS_PREFIX=/opt/cgns
ENV SURROGATE_NEWTON_SOLVER_LOCK=/opt/build-context/solver-stack.lock.yaml
ENV LD_LIBRARY_PATH=/opt/cgns/lib:${CONDA_ENV}/lib

FROM builder-base AS solver-source-git
RUN scripts/fetch_solver_stack.sh /opt/solver-stack --source-mode git

FROM builder-base AS solver-source-bundle
RUN --network=none \
    --mount=type=bind,from=solver-bundles,source=.,target=/opt/solver-bundles,readonly \
    scripts/fetch_solver_stack.sh /opt/solver-stack \
        --source-mode bundle \
        --bundle-dir /opt/solver-bundles

FROM solver-source-${SOLVER_SOURCE_MODE} AS builder
RUN scripts/build_solver_stack.sh /opt/solver-stack \
    && scripts/verify_solver_stack.sh /opt/solver-stack

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN mkdir -p \
        /opt/solver-export/cgnsutilities \
        /opt/solver-export/pyhyp \
        /opt/solver-export/adflow \
    && git -C /opt/solver-stack/cgnsutilities archive HEAD \
        LICENSE.md README.md cgnsutilities \
        | tar -x -C /opt/solver-export/cgnsutilities \
    && git -C /opt/solver-stack/pyhyp archive HEAD \
        LICENSE.md README.md SURROGATE_NEWTON_CHANGES.md pyhyp \
        | tar -x -C /opt/solver-export/pyhyp \
    && git -C /opt/solver-stack/adflow archive HEAD \
        LICENSE.md README.md SURROGATE_NEWTON_CHANGES.md adflow src/f2py \
        | tar -x -C /opt/solver-export/adflow \
    && cp \
        /opt/solver-stack/cgnsutilities/cgnsutilities/libcgns_utils.so \
        /opt/solver-export/cgnsutilities/cgnsutilities/libcgns_utils.so \
    && cp /opt/solver-stack/pyhyp/pyhyp/hyp.so /opt/solver-export/pyhyp/pyhyp/hyp.so \
    && cp /opt/solver-stack/adflow/adflow/libadflow.so /opt/solver-export/adflow/adflow/libadflow.so \
    && git -C /opt/solver-stack/cgnsutilities rev-parse HEAD \
        > /opt/solver-export/cgnsutilities/.surrogate-newton-revision \
    && git -C /opt/solver-stack/pyhyp rev-parse HEAD > /opt/solver-export/pyhyp/.surrogate-newton-revision \
    && git -C /opt/solver-stack/adflow rev-parse HEAD > /opt/solver-export/adflow/.surrogate-newton-revision

COPY . /opt/surrogate-newton-source
RUN ${CONDA_ENV}/bin/python -m pip wheel \
    --no-deps --no-build-isolation \
    --wheel-dir /opt/wheels \
    /opt/surrogate-newton-source

FROM runtime-os AS runtime

ARG VCS_REF=unknown
ARG IMAGE_VERSION=0.1.0-rc1
ARG SOURCE_URL=https://github.com/LeiMingcheng/surrogate-newton-cfd

LABEL org.opencontainers.image.title="Surrogate-Newton CFD runtime" \
      org.opencontainers.image.description="RAE2822 surrogate inference and ADflow Newton-Krylov correction runtime" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.licenses="BSD-3-Clause AND Apache-2.0 AND LGPL-2.1-only"

RUN useradd --create-home --home-dir /home/surrogate --uid 10001 surrogate \
    && mkdir -p /models /runtime/tmp/cfd /opt/surrogate-newton \
    && chown -R surrogate:surrogate /runtime

ENV CONDA_ENV=/opt/conda/envs/surrogate-newton
ENV PATH=${CONDA_ENV}/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/cgns/lib:${CONDA_ENV}/lib
ENV PYTHONNOUSERSITE=1
ENV PYTHONPATH=/opt/solver-stack/cgnsutilities:/opt/solver-stack/pyhyp:/opt/solver-stack/adflow:/opt/surrogate-newton/src
ENV SURROGATE_NEWTON_ADFLOW_ROOT=/opt/solver-stack/adflow
ENV SURROGATE_NEWTON_CGNSUTILITIES_ROOT=/opt/solver-stack/cgnsutilities
ENV SURROGATE_NEWTON_PYTHON=${CONDA_ENV}/bin/python
ENV SURROGATE_NEWTON_MODEL_DIR=/models
ENV SURROGATE_NEWTON_RUNTIME_DIR=/runtime
ENV SURROGATE_NEWTON_RUNTIME_ROOT=/runtime/tmp
ENV CFD_RUNTIME_TMPDIR=/runtime/tmp/cfd
ENV OMP_NUM_THREADS=1

COPY --from=runtime-environment /opt/conda/envs/surrogate-newton ${CONDA_ENV}
COPY --from=builder /opt/cgns /opt/cgns
COPY --from=builder /opt/solver-export /opt/solver-stack
COPY --from=builder /opt/wheels /tmp/wheels
COPY . /opt/surrogate-newton/src

RUN ${CONDA_ENV}/bin/python -m pip install --no-deps /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels \
    && ${CONDA_ENV}/bin/python -m pip check

WORKDIR /opt/surrogate-newton/src
USER surrogate
ENTRYPOINT ["/opt/surrogate-newton/src/deployment/container/entrypoint.sh"]
CMD ["source-smoke"]
