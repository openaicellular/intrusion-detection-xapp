FROM nvidia/cuda:11.1.1-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

# O-RAN upstream repo: release or staging (staging probably
# only makes sense if you also set ORAN_VERSIONS=latest).
ARG ORAN_REPO=release

# O-RAN version-specific package support: set to `latest` to
# get the latest stuff upstream in packagecloud $ORAN_REPO.
ARG ORAN_VERSIONS=
ARG MDCLOG_VERSION=0.1.1-1
ARG RMR_VERSION=4.4.6

RUN apt-get update \
  && apt-get install -y cmake g++ libssl-dev rapidjson-dev git libboost-all-dev \
    ca-certificates curl gnupg apt-transport-https apt-utils libjpeg-dev zlib1g-dev libfreetype6-dev liblcms2-dev \
    pkg-config autoconf libtool libcurl4-openssl-dev \
  && curl -s https://packagecloud.io/install/repositories/o-ran-sc/${ORAN_REPO}/script.deb.sh | os=debian dist=stretch bash  \
  && ( [ "${ORAN_VERSIONS}" = "latest" ] \
      || apt-get install -y \
             mdclog=$MDCLOG_VERSION mdclog-dev=$MDCLOG_VERSION \
	     rmr=$RMR_VERSION rmr-dev=$RMR_VERSION \
      && apt-get install -y \
             mdclog mdclog-dev \
	     rmr rmr-dev \
     ) \
  && rm -rf /var/lib/apt/lists/*
  
RUN apt-get update \
  && apt-get install -y python3-dev python3-pip \
  && python3 -m pip install influxdb numpy datetime psutil \                                     
  && update-alternatives --install /usr/bin/python python /usr/bin/python3 1
  
WORKDIR /app

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN cd /tmp \
  && git clone https://gitlab.flux.utah.edu/powderrenewpublic/xapp-frame-cpp \
  && cd xapp-frame-cpp \
  && mkdir -p build && cd build \
  && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DDEV_PKG=1 .. \
  && make && make install && ldconfig \
  && cd /tmp && rm -rf /tmp/xapp-frame-cpp

RUN cd /tmp \
  && git clone https://gitlab.flux.utah.edu/powderrenewpublic/asn1c-eurecom asn1c \
  && cd asn1c \
  && git checkout f12568d617dbf48497588f8e227d70388fa217c9 \
  && autoreconf -iv \
  && ./configure \
  && make install \
  && ldconfig \
  && cd .. \
  && rm -rf /tmp/asn1c

RUN cd /tmp \
  && git clone https://gitlab.flux.utah.edu/powderrenewpublic/pistache \
  && cd pistache && mkdir build && cd build \
  && cmake ../ && make install && ldconfig \
  && cd .. && rm -rf /tmp/pistache

RUN cd /tmp \
  && git clone https://github.com/offa/influxdb-cxx \
  && cd influxdb-cxx \
  && git checkout 6b76bd02f26166e03888214914e5f9a000feb7d8 \
  && mkdir -p build && cd build \
  && cmake ../ && make install && ldconfig \
  && cd .. && rm -rf /tmp/influxdb-cxx
    
COPY . /nexran

RUN chown -R root:root /nexran
RUN chmod -R 755 /nexran

RUN cd /nexran \
  && rm -rf build && mkdir build && cd build \
  && ( [ ! -e /nexran/lib/e2ap/messages/e2ap-v01.00.asn1 ] \
       && mkdir -p /nexran/lib/e2ap/messages/generated \
       && curl -L https://github.com/openaicellular/srslte-e2/raw/dcb36641c074a9b6d56a3a63712685efbb5ade1e/e2_bindings/E2AP-v01.00-generated-bindings.tar.gz | tar -xzv -C /nexran/lib/e2ap/messages/generated \
       && echo "RIC_GENERATED_E2AP_BINDING_DIR:STRING=/nexran/lib/e2ap/messages/generated/E2AP-v01.00" >> CMakeCache.txt ) \
     || true \
  && ( [ ! -e /nexran/lib/e2sm/messages/e2sm-kpm-v01.00.asn1 ] \
       && mkdir -p /nexran/lib/e2sm/messages/generated \
       && curl -L https://github.com/openaicellular/srslte-e2/raw/dcb36641c074a9b6d56a3a63712685efbb5ade1e/e2_bindings/E2SM-KPM-ext-generated-bindings.tar.gz | tar -xzv -C /nexran/lib/e2sm/messages/generated \
       && cp /nexran/E2SM_KPM_PerUEReportListItem.h /nexran/lib/e2sm/messages/generated/E2SM-KPM/E2SM_KPM_PerUEReportListItem.h \
       && cp /nexran/E2SM_KPM_PerUEReportListItem.c /nexran/lib/e2sm/messages/generated/E2SM-KPM/E2SM_KPM_PerUEReportListItem.c \
       && echo "RIC_GENERATED_E2SM_KPM_BINDING_DIR:STRING=/nexran/lib/e2sm/messages/generated/E2SM-KPM" >> CMakeCache.txt ) \
     || true \
  && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ../ \
  && make install && ldconfig

ENV RMR_RTG_SVC="9999" \
    RMR_SEED_RT="/nexran/etc/routes.txt" \
    DEBUG=1 \
    XAPP_NAME="nexran" \
    XAPP_ID="1"

CMD [ "/usr/local/bin/nexran" ]

EXPOSE 8000

