#!/bin/bash

####
# Utils
####
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

function echo_error {
    echo -e "${RED}[ERROR]${NC} $1"
}

function echo_warning {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

function echo_info {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# 1. Get options

## Defaults
APPLY="false"

while [[ $# -gt 0 ]]; do
  case $1 in
    -A|--apply)
      APPLY="true"
      shift # past argument
      ;;
    --bt-version)
      BT_VERSION="$2"
      shift # past argument
      shift # past value
      ;;
    --cu-version)
      CU_VERSION="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

# Check if BT_VERSION and CU_VERSION are provided
if [[ -z $BT_VERSION ]]; then
    echo_error "BT_VERSION not provided. Use --bt-version to specify it."
    exit 1
fi

if [[ -z $CU_VERSION ]]; then
    echo_error "CU_VERSION not provided. Use --cu-version to specify it."
    exit 1
fi

DOCKER_IMAGE_NAME="opentensorfdn/bittensor:$BT_VERSION-cubit$CU_VERSION"

# 2. Login
if [[ $APPLY == "true" ]]; then
  echo_info "Docker registry login"
  sudo docker login
else
  echo_warning "Dry run execution. Not login into docker registry"
fi

# 3. Creating docker image
if [[ $APPLY == "true" ]]; then
  echo_info "Building docker image"
else
  echo_warning "Dry run execution. Building docker image '$DOCKER_IMAGE_NAME' but not pushing it"
fi

sudo docker build -t $DOCKER_IMAGE_NAME -f ./docker/Dockerfile .

# 4. Uploading docker image
if [[ $APPLY == "true" ]]; then
  echo_info "Pushing docker image"
  sudo docker push $DOCKER_IMAGE_NAME
else
  echo_warning "Dry run execution. Not pushing docker image '$DOCKER_IMAGE_NAME'"
fi
