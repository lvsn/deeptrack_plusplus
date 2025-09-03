nvidia-docker run -it \
    --privileged --ipc=host\
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /gel/usr/magar220/raid/Outputs/tracking:/root/outputs \
    -v /gel/usr/magar220/raid/source:/root/source \
    -v /gel/usr/magar220/raid/Datasets:/root/dataset_raid \
    -v /gel/usr/magar220/Datasets:/root/dataset_ssd \
    tracking /bin/bash
