#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 1;

class PoolAvg{
public:
    __aicore__ inline PoolAvg(){}  // Initialize memory
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR poolSize, GM_ADDR y, uint32_t matW, uint32_t matH,
                                uint32_t iptDataNum, uint32_t coreNum){
        poolGm.SetGlobalBuffer((__gm__ DTYPE_POOLSIZE*)poolSize, 1);
        this->poolSize = poolGm.GetValue(0);
        this->fpSize = (float)poolGm.GetValue(0);
        CalcParam(matW, matH, iptDataNum, coreNum);
        
        uint32_t offsetX;
        uint32_t offsetY;
        if(AscendC::GetBlockIdx() < this->bigCoreNum){
            this->poolNum = this->smallPoolNum + 1;
            this->coreDtLenX = (this->smallPoolNum + 1) * this->poolSize;
            this->coreDtLenY = this->smallPoolNum + 1;
            offsetX = AscendC::GetBlockIdx() * (this->smallPoolNum + 1) * this->poolSize;
            offsetY = AscendC::GetBlockIdx() * (this->smallPoolNum + 1);
        }
        else{
            this->poolNum = this->smallPoolNum;
            this->coreDtLenX = this->smallPoolNum * this->poolSize;
            this->coreDtLenY = this->smallPoolNum;
            offsetX = AscendC::GetBlockIdx() * (this->smallPoolNum + 1) * this->poolSize - 
                                (AscendC::GetBlockIdx() - this->bigCoreNum) * this->poolSize;
            offsetY = AscendC::GetBlockIdx() * (this->smallPoolNum + 1) - 
                                (AscendC::GetBlockIdx() - this->bigCoreNum);
        }
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + offsetX, this->coreDtLenX);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + offsetY, this->coreDtLenY);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->poolSize * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, sizeof(DTYPE_Y)); // before modify: this->coreDtLenY * sizeof(DTYPE_Y)
        pipe.InitBuffer(workCalcBuf, this->poolSize * sizeof(DTYPE_X));
        pipe.InitBuffer(storeBuf, sizeof(DTYPE_X));
    }
    __aicore__ inline void Process(){
        //AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        float tmp = 1 / this->fpSize;
        for(uint32_t i = 0; i < this->poolNum; i++){
            avg = 0;
            //CopyIn(i);
            ComputeAvg(i, tmp);
            //yGm.SetValue(i, avg);
            //AscendC::DataCacheCleanAndInvalid<DTYPE_Y, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(yGm);
        }
        AscendC::DataCacheCleanAndInvalid<DTYPE_Y, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(yGm);
        //CopyOut();
    }
private:
    __aicore__ inline void CalcParam(uint32_t matW, uint32_t matH, uint32_t iptDataNum, uint32_t coreNum){
        uint32_t totalPool = iptDataNum / this->poolSize;
        this->bigCoreNum = totalPool % coreNum;
        this->smallPoolNum = totalPool / coreNum;
    }
    __aicore__ inline void CopyIn(uint32_t i){
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[i * this->poolSize], this->poolSize);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void ComputeAvg(uint32_t i, float scalar){
        //AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        //AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        //AscendC::LocalTensor<DTYPE_X> workBuf = workCalcBuf.Get<DTYPE_X>();
        //AscendC::LocalTensor<DTYPE_X> sBuf = storeBuf.Get<DTYPE_X>();
        //AscendC::ReduceSum(yLocal, xLocal, workBuf, this->poolSize);
        //float result = sBuf.GetValue(0);
        //result = result  * scalar;
        float t1, t2;
        for(uint32_t j = 0; j < this->poolSize; j++){
            t1 = xGm.GetValue(i * this->poolSize + j);
            t2 = avg;
            t1 = t1 + t2;
            avg = t1;
        }
        avg = avg * scalar;
        yGm.SetValue(i, avg);
        //AscendC::DataCacheCleanAndInvalid<DTYPE_Y, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(yGm);
        //inQueueX.FreeTensor(xLocal);
    }
    /*__aicore__ inline void CopyOut(){
        //AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm, yLocal, this->coreDtLenY);
        outQueueY.FreeTensor(yLocal);
    }*/
    // create memory management
    AscendC::TPipe pipe;
    // create queue and buffer management
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workCalcBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storeBuf;
    // create GM address management
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_POOLSIZE> poolGm;
    // menber var
    DTYPE_Y avg;
    uint32_t poolSize;
    float fpSize;
    uint32_t bigCoreNum;
    uint32_t smallPoolNum;
    uint32_t poolNum;
    uint32_t coreDtLenX;
    uint32_t coreDtLenY;
};

extern "C" __global__ __aicore__ void pool_avg(GM_ADDR x, GM_ADDR poolSize, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    PoolAvg op;
    op.Init(x, poolSize, y, tiling_data.matW, tiling_data.matH, tiling_data.iptDataNum, tiling_data.coreNum);
    op.Process();
}
