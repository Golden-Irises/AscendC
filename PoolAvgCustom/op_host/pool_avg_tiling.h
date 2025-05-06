
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PoolAvgTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, matH);
  TILING_DATA_FIELD_DEF(uint32_t, matW);
  TILING_DATA_FIELD_DEF(uint32_t, iptDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, coreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PoolAvg, PoolAvgTilingData)
}
