#!/bin/bash
docker run --gpus all --mount type=bind,source=/home/nikhil/ece590hineman,target=/host -itd -p 8899:8899 spinup-nikhil /bin/bash -c "/opt/conda/bin/jupyter notebook --allow-root --notebook-dir=/host --ip='*' --port=8899 --no-browser"

# Useful commands
#  docker container ls -a
#  docker exec -it 2c1879f96912 jupyter notebook list