#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 2;

class VecAdd{
public:
    __aicore__ inline VecAdd(){}        //Initialize memory
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t bigCoreNum, uint32_t bigTotalLen,
                                uint32_t bigTailNum, uint32_t smallTotalLen, uint32_t smallTailNum, uint32_t tileNum, uint32_t tileDataNum){
        uint32_t offset = bigTotalLen * AscendC::GetBlockIdx();
        this->tileNum = tileNum;
        this->tileDataNum = tileDataNum;
        // allocate global memory
        if(AscendC::GetBlockIdx() < bigCoreNum){
            this->tailDataNum = bigTailNum;
            xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + offset, bigTotalLen);
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + offset, bigTotalLen);
            zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + offset, bigTotalLen);
        }
        else{
            this->tailDataNum = smallTailNum;
            xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + offset - 8 * (AscendC::GetBlockIdx()- bigCoreNum), smallTotalLen);
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + offset - 8 * (AscendC::GetBlockIdx()- bigCoreNum), smallTotalLen);
            zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + offset - 8 * (AscendC::GetBlockIdx()- bigCoreNum), smallTotalLen);
        }
	AscendC::printf("TILEDATANUM IS: %d\n", this->tileDataNum);
	AscendC::printf("TAILDATANUM IS: %d\n", this->tailDataNum);
        // allocate memory to TQue obj
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_Z));
    }
    __aicore__ inline void Process(){
        int32_t loopCount = this->tileNum;
        this->transferDataNum = this->tileDataNum;
        for(int32_t i = 0; i < loopCount; i++){
            if(i == loopCount - 1){ this->transferDataNum = this->tailDataNum; }
	    AscendC::printf("%s\n", "READY TO CPOY IN");
            CopyIn(i);
	    AscendC::printf("%s\n", "READY TO COMPUTE");
            Compute(i);
	    AscendC::printf("%s\n", "READY TO COPY OUT");
            CopyOut(i);
        }
    }
private:
    __aicore__ inline void CopyIn(int32_t progress){
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->transferDataNum); // what, where, how many
        AscendC::DataCopy(yLocal, yGm[progress * this->tileDataNum], this->transferDataNum);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress){
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::Add(zLocal, xLocal, yLocal, this->transferDataNum);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress){
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileDataNum], zLocal, this->transferDataNum);
        outQueueZ.FreeTensor(zLocal);
    }
private:
    // memory management obj
    AscendC::TPipe pipe;
    // queue management obj
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    // GM address management obj
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    // member var
    int32_t  tileNum;
    uint32_t tileDataNum;
    uint32_t transferDataNum;
    uint32_t tailDataNum;
};

extern "C" __global__ __aicore__ void vec_add(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    VecAdd op;
    AscendC::printf("%s\n", "READY TO INITIALIZE");
    op.Init(x, y, z, tiling_data.bigCoreNum, tiling_data.bigTotalLen, tiling_data.bigTailNum,
            tiling_data.smallTotalLen, tiling_data.smallTailNum, tiling_data.tileNum, tiling_data.tileDataNum);
    AscendC::printf("%s\n", "INITIALIZE SUCCESS!" );
    op.Process();
}

