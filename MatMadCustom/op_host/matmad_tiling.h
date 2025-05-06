
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

// TCudeTiling struct need include tiling_api.h

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmadTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Matmad, MatmadTilingData)
}
