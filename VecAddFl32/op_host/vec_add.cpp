
#include "vec_add_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t iptTypeSize = 4;     // float type : 4bytes
constexpr uint32_t BUFFER_NUM = 2;      // double buffer
constexpr uint32_t ubNum = 3;           // 2 ipt and 1 opt, 2 more for data type conversion

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    VecAddTilingData tiling;
    auto acdcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());     // get platform info
    uint32_t coreNum = acdcPlatform.GetCoreNumAiv();                                       // get vector core num
    coreNum = 8; //use 8 cores
    // numbers of input data = c->select first input shape->input storage dimensions(if 2D, [a, b]).mul of dim(a*b)
    uint32_t iptNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();  
    uint64_t ubSize = 1536;  // use 1536 bytes
   // acdcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);                // size of UB buffer (byte)
    // max processing data amount(maxTileData) in single processing procedure
    uint32_t maxBlkNum = ubSize / BLOCK_SIZE / BUFFER_NUM / ubNum;   // max block num for every variables
    uint32_t blkDataNum = BLOCK_SIZE / iptTypeSize;                  // data num in one block
    uint32_t tileDataNum = maxBlkNum * blkDataNum;  
    uint32_t iptAlignBlkNum = (iptNum * iptTypeSize + BLOCK_SIZE - 1) / BLOCK_SIZE;     // input block num after align
    // infer core usage, coreNum can not be 0
    uint32_t calcTime = (iptAlignBlkNum + maxBlkNum - 1) / maxBlkNum;
    coreNum = (iptAlignBlkNum < coreNum) ? iptAlignBlkNum : coreNum;
    coreNum = (coreNum > 0) ? coreNum : 1;
    // calc big core info and small core info. look out for the order of mul and div!!!
    int32_t  tileNum = (calcTime + coreNum - 1) / coreNum;     // if iptNum = 0; uint32_t type will encounter error
    uint32_t bigCoreNum = iptAlignBlkNum % coreNum;
    uint32_t smallTotalLen = iptAlignBlkNum / coreNum * blkDataNum;
    uint32_t bigTotalLen = smallTotalLen + blkDataNum;
    uint32_t smallTailNum = smallTotalLen - (tileNum - 1) * tileDataNum;
    uint32_t bigTailNum = smallTailNum + blkDataNum;
    std::cout << "INPUTBLKNUM: " << iptAlignBlkNum << std::endl; // ------DEBUG------
    std::cout << "TILEDATANUM: " << tileDataNum << std::endl; // ------DEBUG------
    std::cout << "CALCTIME: " << calcTime << std::endl; // ------DEBUG------
    std::cout << "TILENUM: " << tileNum << std::endl; // ------DEBUG------
    std::cout << "BIGCORENUM: " << bigCoreNum << std::endl; // ------DEBUG------
    std::cout << "SMALLTOTALLEN: " << smallTotalLen << std::endl; // ------DEBUG------
    std::cout << "BIGTOTALLEN: " << bigTotalLen << std::endl; // ------DEBUG------
    std::cout << "SMALLTAILNUM: " << smallTailNum << std::endl; // ------DEBUG------
    std::cout << "BIGTAILNUM: " << bigTailNum << std::endl; // ------DEBUG------

    context->SetBlockDim(coreNum);
    // assign values to tiling data
    tiling.set_bigCoreNum(bigCoreNum);
    tiling.set_bigTotalLen(bigTotalLen);
    tiling.set_bigTailNum(bigTailNum);
    tiling.set_smallTotalLen(smallTotalLen);
    tiling.set_smallTailNum(smallTailNum);
    tiling.set_tileNum(tileNum);
    tiling.set_tileDataNum(tileDataNum);
    // packing tiling data
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
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
class VecAdd : public OpDef {
public:
    explicit VecAdd(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore()
            .AddConfig("ascend910b");

    }
};

OP_ADD(VecAdd);
}
