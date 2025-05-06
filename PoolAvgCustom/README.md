##PoolAvg算子介绍：
 - 本算子基于AscendC编程语言编写，用于快速处理图片池化过程
 - 用户需要输入经过Im2Col处理的二维矩阵
 - 需要运行在搭载昇腾NPU的设备上

##环境依赖：
 - 请参考官方文档: [CANN安装环境依赖项]（https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/softwareinst/instg/instg_0007.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit）

##算子描述：
1. 计算方法：*y*(float32) = PoolAvg(*x*(float32))
  - 若想更改数据类型，需要对op_host/vec_add.cpp中算子输入输出定义部分进行修改
  - 在op_kernel/vec_add.cpp中采用了GlobalTensor的自动数据类型萃取
2. 输入变量：
 - x: float32类型，尺寸任意的二维矩阵，用户需确保输入尺寸正确
 - poolSize: int32类型，池化窗口尺寸(例如：2x2池化时，该值为4)
3. 输出：
 - y: float32类型，输出数据格式为NHWC，若想转换为一般格式(NCHW)需要进行转置

##算子调测流程：
1. 确保以下环境变量被正确配置
 - `source {your cann install path}/Ascend/ascend-toolkit/set_env.sh`
 - `export PATH={your cmake install path}/bin:$PATH`
 - `export ASCEND_CUSTOM_PATH={your cann install path}/Ascend/ascend-toolkit/latest`
 - `export ASCEND_HOME_DIR={your cann install path}/Ascend/ascend-toolkit/latest`
 - `DDK_PATH={your cann install path}/Ascend/ascend-toolkit/latest`
 - `export NPU_HOST_LIB={your cann install path}/Ascend/ascend-toolkit/latest/arm64-linux/devlib`
2. 编译算子
 - 进入算子文件目录，运行`bash build.sh`开始编译算子，编译完成后提示SUCCESS并生成==build_out==文件夹
3. 测试算子功能
 - 使用==msopst==指令对算子进行测试，测试方法请参考: [算子测试官方文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/devaids/opdev/optool/atlasopdev_16_0029.html)
4. 测试结果样例(尺寸为4x12的输入矩阵，池化窗口为2x2)
![PoolAvg结果样例](./poolavg_result.png)

##遇到的一些问题：
1. 尝试对LocalTensor使用GetValue()获取值失败，故使用了效率较低的方案，从GlobalTensor直接获取值
2. 多核通过DataCacheCleanAndInvalid()方法同步GlobalMemory时可能产生冲突，导致写入失败，故本样例仅采用了1个AICore