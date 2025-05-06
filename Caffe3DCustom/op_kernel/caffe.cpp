#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_SIZE = 8;  // minimum adress alignment is 32 bytes (8 float)

class Caffe{
public:
    __aicore__ inline Caffe(){}  //Initialize memory
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR kernel, GM_ADDR stride, GM_ADDR y, uint32_t H, uint32_t W,
                                uint32_t C, uint32_t krnSizeH, uint32_t krnSizeW, uint32_t kernelSize, uint32_t matSize,
                                uint32_t colNum, uint32_t coreNum){
        strideGm.SetGlobalBuffer((__gm__ DTYPE_STRIDE*)stride, 1);
        this->stride = strideGm.GetValue(0);
        CalcTilingParam(H, W, C, krnSizeH, krnSizeW, coreNum);
        this->colNum = colNum;
        this->kernelSize = kernelSize;
        this->kernelH = krnSizeH;
        this->kernelW = krnSizeW;
        this->matSize = matSize;
        this->matW = W;
        this->matC = C;

        uint32_t offset = AscendC::GetBlockIdx() * (this->smallRowNum + 1) * colNum;
        if(AscendC::GetBlockIdx() < this->bigCoreNum){
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + offset, (this->smallRowNum + 1) * colNum);
        }
        else{
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + offset - colNum * (AscendC::GetBlockIdx() - this->bigCoreNum), this->smallRowNum * colNum);
        }
        // in and out queue size cannot be over than UB memory size
        pipe.InitBuffer(outQueueY, BUFFER_NUM, colNum * sizeof(DTYPE_Y));  // automatically align to 32 bytes
    }
    __aicore__ inline void Process(){
        if(AscendC::GetBlockIdx() < this->bigCoreNum){
            int32_t rowIdx = AscendC::GetBlockIdx() * (this->smallRowNum + 1);  // row index of output
            for(int32_t i = 0; i <= this->smallRowNum; i++){
                int32_t matRowIdx = (rowIdx + i) / (this->rowStep + 1);
                int32_t matColIdx = (rowIdx + i) % (this->rowStep + 1);  // index of origin input
                int32_t index = matRowIdx * this->stride * this->matW + matColIdx * this->stride;
                Im2Col(i, index);
                // CopyOut(i);
            }
            AscendC::DataCacheCleanAndInvalid<DTYPE_Y, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(yGm);
        }
        else{
            int32_t rowIdx = AscendC::GetBlockIdx() * (this->smallRowNum + 1) - (AscendC::GetBlockIdx() - 
                             this->bigCoreNum);
            for(int32_t i = 0; i < this->smallRowNum; i++){
                int32_t matRowIdx = (rowIdx + i) / (this->rowStep + 1);
                int32_t matColIdx = (rowIdx + i) % (this->rowStep + 1);  // index of origin input
                int32_t index = matRowIdx * this->stride * this->matW + matColIdx * this->stride;
                Im2Col(i, index);
                // CopyOut(i);
            }
            AscendC::DataCacheCleanAndInvalid<DTYPE_Y, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(yGm);
        }
    }
private:
    __aicore__ inline void CalcTilingParam(uint32_t H, uint32_t W, uint32_t C, uint32_t krnSizeH, uint32_t krnSizeW,
                                            uint32_t coreNum){
        this->rowStep = (W - krnSizeW) / this->stride;
        this->colStep = (H - krnSizeH) / this->stride;
        uint32_t rowNum = (this->rowStep + 1) * (this->colStep + 1);
        this->bigCoreNum = rowNum % coreNum;
        this->smallRowNum = rowNum / coreNum;
    }
    __aicore__ inline void Im2Col(int32_t progress, int32_t index){
        // AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        for(int32_t i = 0; i < this->matC; i++){
            for(int32_t j = 0; j < this->kernelH; j++){
                for(int32_t k = 0; k < this->kernelW; k++){
                    yGm.SetValue(progress*this->kernelSize*this->matC+i*this->kernelSize+j*this->kernelW+k, index + i * this->matSize + j * this->matW + k);
                }
            }
        }
        // outQueueY.EnQue<DTYPE_Y>(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress){
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->colNum], yLocal, this->colNum);  // use DataCopy between GM and buffer
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
    uint32_t colNum;
    uint32_t bigCoreNum;
    uint32_t smallRowNum;
    uint32_t kernelSize;
    uint32_t kernelH;
    uint32_t kernelW;
    uint32_t matSize;
    uint32_t matW;
    uint32_t matC;
};

extern "C" __global__ __aicore__ void caffe(GM_ADDR x, GM_ADDR kernel, GM_ADDR stride, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    Caffe op;
    op.Init(x, kernel, stride, y, tiling_data.H, tiling_data.W, tiling_data.C, tiling_data.krnSizeH,
            tiling_data.krnSizeW, tiling_data.kernelSize, tiling_data.matSize, tiling_data.colNum, tiling_data.coreNum);
    op.Process();
}
