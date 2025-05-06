#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_SIZE = 8;  // minimum adress alignment is 32 bytes (8 float)

class Caffe2D{
public:
    __aicore__ inline Caffe2D(){}  //Initialize memory
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR kernel, GM_ADDR stride, GM_ADDR y, uint32_t H, uint32_t W,
                                uint32_t krnSizeH, uint32_t krnSizeW, uint32_t kernelSize, uint32_t matSize){
        strideGm.SetGlobalBuffer((__gm__ DTYPE_STRIDE*)stride, 1);
        this->stride = strideGm.GetValue(0);
        CalcTilingParam(H, W, krnSizeH, krnSizeW);
        this->kernelSize = kernelSize;
        this->kernelH = krnSizeH;
        this->kernelW = krnSizeW;
        this->matSize = matSize;
        this->matW = W;

        uint32_t offset = AscendC::GetBlockIdx() * (this->smallRowNum + 1) * kernelSize;
        if(AscendC::GetBlockIdx() < this->bigCoreNum){
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + offset, (this->smallRowNum + 1) * kernelSize);
        }
        else{
            //alignLen = (this->smallRowNum * kernelSize + BLOCK_SIZE -1) / BLOCK_SIZE * BLOCK_SIZE;
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + offset - kernelSize * (AscendC::GetBlockIdx() - this->bigCoreNum), this->smallRowNum * kernelSize);
            //yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + offset - kernelSize * (AscendC::GetBlockIdx() - this->bigCoreNum), alignLen);
        }
        // in and out queue size cannot be over than UB memory size
        pipe.InitBuffer(outQueueY, BUFFER_NUM, kernelSize * sizeof(DTYPE_Y));  // automatically align to 32 bytes
    }
    __aicore__ inline void Process(){
        if(AscendC::GetBlockIdx() < this->bigCoreNum){
            int32_t rowIdx = AscendC::GetBlockIdx() * (this->smallRowNum + 1);  // row index of output
            for(int32_t i = 0; i <= this->smallRowNum; i++){
                int32_t matRowIdx = (rowIdx + i) / (this->rowStep + 1);
                int32_t matColIdx = (rowIdx + i) % (this->rowStep + 1);  // index of origin input
                int32_t index = matRowIdx * this->stride * this->matW + matColIdx * this->stride;
                CreateIdx(index);
                CopyOut(i);
            }
        }
        else{
            int32_t rowIdx = AscendC::GetBlockIdx() * (this->smallRowNum + 1) - (AscendC::GetBlockIdx() - 
                             this->bigCoreNum);
            for(int32_t i = 0; i < this->smallRowNum; i++){
                int32_t matRowIdx = (rowIdx + i) / (this->rowStep + 1);
                int32_t matColIdx = (rowIdx + i) % (this->rowStep + 1);  // index of origin input
                int32_t index = matRowIdx * this->stride * this->matW + matColIdx * this->stride;
                CreateIdx(index);
                CopyOut(i);
            }
        }
    }
private:
    __aicore__ inline void CalcTilingParam(uint32_t H, uint32_t W, uint32_t krnSizeH, uint32_t krnSizeW){
        this->rowStep = (W - krnSizeW) / this->stride;
        this->colStep = (H - krnSizeH) / this->stride;
        uint32_t rowNum = (this->rowStep + 1) * (this->colStep + 1);
        this->bigCoreNum = rowNum % 8;  // if used core < 8, unused core will be idle (waste computing resource)
        this->smallRowNum = rowNum / 8;
    }
    __aicore__ inline void CreateIdx(int32_t index){
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        for(int32_t i = 0; i < kernelH; i++){
            for(int32_t j = 0; j < this->kernelW; j++){
                yLocal.SetValue(i * this->kernelW + j, static_cast<int>(index + i * this->matW + j));
            }
        }
        // yLocal.SetValue(8, 9);
        outQueueY.EnQue<DTYPE_Y>(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress){
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->kernelSize], yLocal, this->kernelSize);  // DataCopy between GM and buffer
        outQueueY.FreeTensor(yLocal);
    }

    // create memory management unit
    AscendC::TPipe pipe;
    // create cache unit
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    // create GM adress management unit
    AscendC::GlobalTensor<DTYPE_STRIDE> strideGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    //member var
    uint32_t stride;
    uint32_t rowStep;
    uint32_t colStep;
    uint32_t bigCoreNum;
    uint32_t smallRowNum;
    uint32_t kernelSize;
    uint32_t kernelH;
    uint32_t kernelW;
    uint32_t matSize;
    uint32_t matW;
};

extern "C" __global__ __aicore__ void caffe2_d(GM_ADDR x, GM_ADDR kernel, GM_ADDR stride, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    Caffe2D op;
    op.Init(x, kernel, stride, y, tiling_data.H, tiling_data.W, tiling_data.krnSizeH, tiling_data.krnSizeW, tiling_data.kernelSize, tiling_data.matSize);
    op.Process();
}
