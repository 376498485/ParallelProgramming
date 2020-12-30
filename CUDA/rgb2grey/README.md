输入命令

export PKG_CONFIG_PATH=/usr/local/lib64/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib64
nvcc pkg-config opencv --cflags rgb2grey.cu  -o rgb2grey pkg-config opencv --libs
./rgb2grey


