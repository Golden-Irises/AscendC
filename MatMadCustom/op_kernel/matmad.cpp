#include "kernel_operator.h"
#include "lib/matmul_intf.h"

class MatMad{
public:
    __aicore__ inline MatMad(){}        //Initialize memory

    // create Matmul obj
    matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
                   matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
                   matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
                   matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>> mm;

    __aicore__ inline void Init(GM_ADDR weight, GM_ADDR x, GM_ADDR bias, GM_ADDR y,
                                GM_ADDR workspace, const TCubeTiling &tiling){
        this->tiling = tiling;
        wGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight), tiling.M * tiling.Ka);
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), tiling.Kb * tiling.N);
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias), tiling.N);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), tiling.M * tiling.N);
        // ensure systtem workspace exist if use Matmul api
        //assert(GetSysWorkSpacePtr == nullptr, "Set system workspace failed!");
        if(GetSysWorkSpacePtr() == nullptr){ return; }
    }
    __aicore__ inline void Process(AscendC::TPipe *pipe){
        mm.SetTensorA(wGm);
        mm.SetTensorB(xGm);
        mm.SetBias(bGm);
        mm.IterateAll(yGm);
        mm.End();
    }

private:
    // GM address management obj
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> yGm;
    // member var
    TCubeTiling tiling;
};

extern "C" __global__ __aicore__ void matmad(GM_ADDR weight, GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    MatMad op;
    AscendC::TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm, &tiling_data.cTiling);  // Initialize matmul obj
    op.Init(weight, x, bias, y, workspace, tiling_data.cTiling);
    op.Process(&pipe);
}
