
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CaffeTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, W);
  TILING_DATA_FIELD_DEF(uint32_t, H);
  TILING_DATA_FIELD_DEF(uint32_t, C);
  TILING_DATA_FIELD_DEF(uint32_t, krnSizeH);
  TILING_DATA_FIELD_DEF(uint32_t, krnSizeW);
  TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
  TILING_DATA_FIELD_DEF(uint32_t, matSize);
  TILING_DATA_FIELD_DEF(uint32_t, colNum);
  TILING_DATA_FIELD_DEF(uint32_t, coreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Caffe, CaffeTilingData)
}
