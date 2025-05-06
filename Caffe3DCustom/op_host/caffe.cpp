
#include "caffe_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    CaffeTilingData tiling;
    auto asdcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto xShape = context->GetInputShape(0)->GetOriginShape();
    auto kShape = context->GetInputShape(1)->GetOriginShape();
    //uint32_t coreNum = asdcPlatform.GetCoreNumAiv();
    //coreNum = (coreNum > 8) ? 8 : coreNum;
    uint32_t coreNum = 1;  // multi-core may cause conflict when writing GM

    uint32_t C = xShape.GetDim(0);
    uint32_t H = xShape.GetDim(1);
    uint32_t W = xShape.GetDim(2);
    uint32_t krnSizeH = kShape.GetDim(0);
    uint32_t krnSizeW = kShape.GetDim(1);
    uint32_t kernelSize = krnSizeH * krnSizeW;
    uint32_t matSize = H * W;
    uint32_t colNum = krnSizeH * krnSizeW * C;
    // Validation
    std::cout << "Matrix_H: " << H << std::endl;
    std::cout << "Matrix_W: " << W << std::endl;
    std::cout << "Matrix_C: " << C << std::endl;
    std::cout << "Matrix_Size: " << matSize << std::endl;
    std::cout << "Kernel_H: " << krnSizeH << std::endl;
    std::cout << "Kernel_W: " << krnSizeW << std::endl;
    std::cout << "Output_ColNum: " << colNum << std::endl;
    std::cout << "UsedCoreNum: " << coreNum << std::endl;
    // Validation End
    tiling.set_H(H);
    tiling.set_W(W);
    tiling.set_C(C);
    tiling.set_krnSizeH(krnSizeH);
    tiling.set_krnSizeW(krnSizeW);
    tiling.set_kernelSize(kernelSize);
    tiling.set_matSize(matSize);
    tiling.set_colNum(colNum);
    tiling.set_coreNum(coreNum);

    context->SetBlockDim(coreNum);  // use 1 aicore
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
    
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DataType::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class Caffe : public OpDef {
public:
    explicit Caffe(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("kernel")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("stride")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b")
            .AddConfig("ascend310b");

    }
};

OP_ADD(Caffe);
}
