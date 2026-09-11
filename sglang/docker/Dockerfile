# Pinned Linux ARM64 dev-dsv41 image; no dependency upgrades.
FROM lmsysorg/sglang@sha256:3475d88ec3124867d9d6f7b3bd49afdcf8ef4d6a9f8454fab331d5adb5204de7
WORKDIR /opt/sglang8
COPY adapter /opt/sglang8/adapter
COPY runtime /opt/sglang8/runtime
COPY versions.lock.json LICENSE LICENSE.sglang LICENSE.upstream-MIT NOTICE /opt/sglang8/
COPY patches /tmp/ep4-patches
COPY scripts/apply-overlays.py /tmp/ep4-apply-overlays.py
RUN python3 /tmp/ep4-apply-overlays.py --source /tmp/ep4-patches --target /sgl-workspace/sglang
ENV PYTHONPATH=/opt/sglang8/adapter:/opt/sglang8/runtime
RUN python3 -m compileall -q /opt/sglang8/adapter /opt/sglang8/runtime
LABEL org.opencontainers.image.description="EP4: eight-Spark DeepSeek V4.1 Flash, native H8 MLA, DSpark verify/index fusion, resident Engram, EP4/MoE-TP2"
HEALTHCHECK NONE
ENTRYPOINT ["python3", "-u", "/opt/sglang8/runtime/launch.py"]
