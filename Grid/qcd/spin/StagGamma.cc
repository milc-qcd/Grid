#include <Grid/Grid.h>

NAMESPACE_BEGIN(Grid);

const std::array<const StagGamma::StagAlgebra, 4> StagGamma::gmu = {{
    StagGamma::StagAlgebra::GX,
    StagGamma::StagAlgebra::GY,
    StagGamma::StagAlgebra::GZ,
    StagGamma::StagAlgebra::GT}};

// XYZT convention
const std::array<const char *, StagGamma::nGamma> StagGamma::name = {{
     "G1" ,
     "GT" ,
     "GZ" ,
     "GZT",
     "GY" ,
     "GYT",
     "GYZ",
     "G5X",
     "GX" ,
     "GXT",
     "GZX",
     "G5Y",
     "GXY",
     "G5Z",
     "G5T",
     "G5" }};

// TXYZ convention
// const std::array<const char *, StagGamma::nGamma> StagGamma::name = {{
//      "G1" ,
//      "GZ" ,
//      "GY" ,
//      "GYZ",
//      "GX" ,
//      "GZX",
//      "GXY",
//      "G5T",
//      "GT" ,
//      "GZT",
//      "GYT",
//      "G5X",
//      "GXT",
//      "G5Y",
//      "G5Z",
//      "G5" }};

NAMESPACE_END(Grid);