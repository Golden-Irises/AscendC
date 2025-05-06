##MatMad算子介绍：
 - 本算子基于AscendC编程语言编写，用于快速处理矩阵乘法
 - 需要运行在搭载昇腾NPU的设备上

##环境依赖：
 - 请参考官方文档: [CANN安装环境依赖项]（https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/softwareinst/instg/instg_0007.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit）

##算子描述：
1. 计算方法：*z*(float32) = *x*(float32) * *y*(float32)
  - 若想更改数据类型，需要对op_host/vec_add.cpp中参数==cubeTiling==结构体中的DataType进行调整，样式为`DT_DATATYPE`
  - 此外还应对op_kernel/vec_add.cpp中涉及数据类型部分进行修改
2. 输入变量：
  - x: float32类型，任意尺寸二维矩阵，尺寸为M x K
  - y: float32类型，任意尺寸二维矩阵，尺寸为K x N
3. 输出：
  - z: float32类型，输出形状为M x N

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
4. 测试结果样例(尺寸为5x12和12x7的两个矩阵相乘)
![MatMad结果样例](./matmad_result.png)