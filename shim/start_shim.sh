export LD_PRELOAD="/usr/lib/$(arch)-linux-gnu/libmimalloc.so.2"

: "${MACHINE_LEARNING_HOST:=0.0.0.0}"
: "${MACHINE_LEARNING_PORT:=3003}"
: "${MACHINE_LEARNING_WORKERS:=1}"
: "${MACHINE_LEARNING_WORKER_TIMEOUT:=120}"

gunicorn shim_host:app -w $MACHINE_LEARNING_WORKERS -b $MACHINE_LEARNING_HOST:$MACHINE_LEARNING_PORT -t $MACHINE_LEARNING_WORKER_TIMEOUT -k uvicorn.workers.UvicornWorker