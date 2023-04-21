### Build docker images
```shell
docker build  -f docker/Dockerfile.base . -t resdsql-base
docker build  -f docker/Dockerfile.infer . -t resdsql

# run the container
docker run -it --rm -p 8000:8000 -v $(pwd)/models:/models --name resdsql resdsql
```