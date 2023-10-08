FROM public.ecr.aws/lambda/python:3.10

ENV TRANSFORMERS_CACHE=${LAMBDA_TASK_ROOT}/cache

RUN yum upgrade -y && yum install python-devel cmake gcc-c++ -y
RUN pip install --upgrade pip && pip install --upgrade setuptools

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install -r requirements.txt

COPY app ${LAMBDA_TASK_ROOT}/app

CMD ["app.inpoint.handler"]