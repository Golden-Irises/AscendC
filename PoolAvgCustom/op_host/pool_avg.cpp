
#include "pool_avg_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    PoolAvgTilingData tiling;
    auto asdcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // uint32_t iptDataNum = context->GetInputTensor(0)->GetShapeSize();
    auto xShape = context->GetInputTensor(0)->GetOriginShape();
    uint32_t matH = xShape.GetDim(0);
    uint32_t matW = xShape.GetDim(1);
    uint32_t iptDataNum = matH * matW;

    // auto dType = context->GetInputTensor(0)->GetDataType();  // get datatype of input
    // uint32_t dTypeLen;
    // if(dType == ge::DT_FLOAT){ dTypeLen = 4; }
    uint32_t coreNum = asdcPlatform.GetCoreNumAiv();
    coreNum = (coreNum > 1) ? 1 : coreNum;  // use 8 aicores at most (Test: use 1 core)

    // Validation
    std::cout << "Matrix_W: " << matW << std::endl;
    std::cout << "Matrix_H: " << matH << std::endl;
    //std::cout << "InputDataNum: " << iptDataNum << std::endl;
    std::cout << "UseCoreNum: " << coreNum << std::endl;
    // Validation End

    context->SetBlockDim(coreNum);
    tiling.set_matW(matW);
    tiling.set_matH(matH);
    tiling.set_iptDataNum(iptDataNum);
    tiling.set_coreNum(coreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    // *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class PoolAvg : public OpDef {
public:
    explicit PoolAvg(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("poolSize")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b")
            .AddConfig("ascend310b");

    }
};

OP_ADD(PoolAvg);
}
