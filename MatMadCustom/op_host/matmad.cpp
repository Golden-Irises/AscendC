
#include "matmad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // get matrix params
    auto wShape = context->GetInputTensor(0)->GetOriginShape();
    auto xShape = context->GetInputTensor(1)->GetOriginShape();
    uint32_t M = wShape.GetDim(0);
    uint32_t K = wShape.GetDim(1);
    uint32_t N = xShape.GetDim(1);

    auto asdcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // call Matmul Tiling API: create a single core tiling obj
    matmul_tiling::MatmulApiTiling cubeTiling(asdcPlatform);
    // set matrix params
    cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetShape(M, N, K);
    cubeTiling.SetOrgShape(M, N, K);
    cubeTiling.SetFixSplit(-1, -1, -1);  // baseK cannot be set in CANN ver.8.0.0
    cubeTiling.EnableBias(true);
    cubeTiling.SetBufferSpace(-1, -1, -1);  // max L1, L0C, UB buffer on aicore
    //cubeTiling.SetDoubleBuffer(a, b, c, bias, transND2NZ, transNZ2ND);  not support in CANN ver.8.0.0

    MatmadTilingData tiling;  // refer from matmad_tiling.h
    // get tiling data to TCubeTiling struct, if failed, return value is -1
    if(cubeTiling.GetTiling(tiling.cTiling) == -1){
        return ge::GRAPH_FAILED;
    }
    std::cout << "M: " << M << std::endl;  //-----DEBUG-----
    std::cout << "N: " << N << std::endl;  //-----DEBUG-----
    std::cout << "K: " << K << std::endl;  //-----DEBUG-----

    context->SetBlockDim(1);  // only 1 aicore on OrangePi AIpro
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize()); 
    // set total workspace size
    size_t usrWorkspaceSize = 0;
    size_t sysWorkspaceSize = static_cast<size_t>(asdcPlatform.GetLibApiWorkSpaceSize());  // get workspace api need
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrWorkspaceSize + sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Matmad : public OpDef {
public:
    explicit Matmad(const char* name) : OpDef(name)
    {
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore()
            .SetTiling(optiling::TilingFunc)
	    .AddConfig("ascend310b")
            .AddConfig("ascend910b");

    }
};

OP_ADD(Matmad);
}
