
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(VecAddTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);     // if block_num % core_num != 0, some core process 1 more block
  TILING_DATA_FIELD_DEF(uint32_t, bigTotalLen);    // total data number processed in big core (used for allocate memory)
  TILING_DATA_FIELD_DEF(uint32_t, bigTailNum);     // tail data number processed in big core
  TILING_DATA_FIELD_DEF(uint32_t, smallTotalLen);
  TILING_DATA_FIELD_DEF(uint32_t, smallTailNum);
  TILING_DATA_FIELD_DEF(int32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(VecAdd, VecAddTilingData)
}
