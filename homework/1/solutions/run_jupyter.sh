#!/bin/bash
docker run --mount type=bind,source=/home/nikhil/Courses/ece590hineman,target=/host -itd -p 8899:8899 spinup-nikhil /bin/bash -c "/opt/conda/bin/jupyter notebook --allow-root --notebook-dir=/host --ip='*' --port=8899 --no-browser"
