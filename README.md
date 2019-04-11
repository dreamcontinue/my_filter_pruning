#####original sfp
1. init model
2. **sfp mask**
3. train n epochs
4. sfp mask
5. 3->4
####Q
mask只有在前面的几轮会更新，之后将基本保持不变（100轮后根本不变）(resnet32 0.7)
每次训练出来的精度差别较大（resnet32 0.7 相差1%多）

#####original my
1. init model
2. **my mask**
3. train n epochs
4. my mask
5. 3->4
####Q
mask的0和实际模型包含的0不一致 修剪比率为0.7时不如sfp 0.9时十分接近但flops小


#####modify my 1
1. init model
2. **sfp mask**
3. train n epochs
4. my mask
5. 3->4
####Q
最终mask的大小直接趋近于sfp

#####modify my 2
1. init model
2. **sfp mask**
3. train n epochs
4. fisrt sfp mask
5. second mask col
6. thrid mask
5. 3->6
####Q
效果差
#####modify my 3
1. init model
2. train n epochs
3. my mask
4. 2->3
####Q
效果也差


