# SuperPoint

## compile
```shell
mkdir build
cd build
cmake ..
make
```

## Run
### test monocular image
```shell
./Superpoint_a311d $model $image_dir
```

### test stereo
normal test    
```shell
# $dirname is left and right parent dir;
./Superpoint_a311d $model $dirname stereo
```

save every line for math     
```shell
# $dirname is left and right parent dir;
./Superpoint_a311d $model $dirname stereo saveline
```

### test sequence    


time is too long, since filter the image pair with large difference.    

normal test      
```shell
# $dirname is parent dir of images;
./Superpoint_a311d $model $dirname sequence
```

save every line for math
```shell
# $dirname is parent dir of images;
./Superpoint_a311d $model $dirname sequence saveline
```



## model

| 说明             | 地址 |
|----------------| --- |
| official model | sftp://192.168.50.55/data/MODEL/relocation/superpoint/official/superpoint_v1.nb|
