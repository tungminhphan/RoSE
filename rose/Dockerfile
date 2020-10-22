FROM ubuntu:18.04

# set a directory for the app
WORKDIR /usr/src/code

# copy all the files to the container
COPY . .

# install dependencies

#RUN pip install --no-cache-dir -r requirements.txt
# currently installs python3 
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && pip3 install numpy scipy pandas matplotlib ipdb networkx

# tell the port number the container should expose
#EXPOSE 5000

# run the command
#CMD ["python3", "./rose.py"]
