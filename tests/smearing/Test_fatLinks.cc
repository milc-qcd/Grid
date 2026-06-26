/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./tests/smearing/Test_fatLinks.cc

Copyright (C) 2023

Author: D. A. Clarke <clarke.davida@gmail.com>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

See the full license in the file "LICENSE" in the top level distribution
directory
*************************************************************************************/
/*  END LEGAL */
/*
    @file Test_fatLinks.cc
    @brief Compare input fat / long links against the HISQ smearing of the
           input gauge configuration and report the residual.

           The two-level HISQ smearing (fat7 -> U(3) project -> asqtad + Naik)
           is applied to the gauge link, producing a smeared fat link and a
           smeared long (Naik) link. These are compared directly, link by link,
           against the supplied fat / long links, and the relative residual

               || U^input  -  U^smeared || / || U^smeared ||

           is reported for both the fat and the long links.

    Usage:
        Test_fatLinks --gauge U.ildg --fat fat.ildg --long long.ildg \
                      [--tol 1e-8] \
                      --grid 8.8.8.8 --mpi 1.1.1.1

    Any of --gauge / --fat / --long may be omitted:
      * no --gauge      -> a random hot gauge configuration is generated;
      * no --fat/--long -> the HISQ-smeared links are reused as the inputs
        (smoke mode: the residual MUST be zero).
*/

#include <Grid/Grid.h>
#include <Grid/parallelIO/IldgIO.h> // IldgReader (needs --with-lime / HAVE_LIME)
#include <Grid/parallelIO/NerscIO.h>    // NerscIO (always available)
#include <Grid/qcd/smearing/Smearing.h> // pulls HighlyImprovedStaggeredFermionImpl
#include <cassert>
#include <cmath>
using namespace Grid;

//
// small helper: parse a RealD valued option, with a default when absent
//
static RealD getRealOption(char **argv, int argc, const std::string &opt,
                           RealD def) {
  std::string s = GridCmdOptionPayload(argv, argv + argc, opt);
  if (s.empty())
    return def;
  try {
    return std::stod(s);
  } catch (...) {
    std::cout << GridLogMessage << "Could not parse " << opt << " '" << s
              << "'; using default " << def << std::endl;
    return def;
  }
}

//
// small helper: read a gauge field from file. Uses ILDG when the library
// was built with LIME support (HAVE_LIME), otherwise falls back to Nersc.
//
static void readConfig(LatticeGaugeField &U, const std::string &file,
                       const char *label) {
  FieldMetaData header;
#ifdef HAVE_LIME
  IldgReader IR;
  IR.open(file);
  IR.readConfiguration(U, header);
  IR.close();
  std::cout << GridLogMessage << "Read " << label << " (ILDG) from " << file
            << " (plaquette " << header.plaquette << ")" << std::endl;
#else
  NerscIO::readConfiguration(U, header, file);
  std::cout << GridLogMessage << "Read " << label << " (Nersc) from " << file
            << " (plaquette " << header.plaquette << ")" << std::endl;
#endif
}

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  ////////////////////////////////////////////////////////////////////////////
  // Command-line arguments
  ////////////////////////////////////////////////////////////////////////////
  std::string gaugeFile = GridCmdOptionPayload(argv, argv + argc, "--gauge");
  std::string fatFile = GridCmdOptionPayload(argv, argv + argc, "--fat");
  std::string longFile = GridCmdOptionPayload(argv, argv + argc, "--long");
  RealD tol = getRealOption(argv, argc, "--tol", 1e-8);

  Coordinate latt_size = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd, vComplexD::Nsimd());
  Coordinate mpi_layout = GridDefaultMpi();
  GridCartesian Grid(latt_size, simd_layout, mpi_layout);

  std::vector<int> seeds({1, 2, 3, 4});
  GridParallelRNG pRNG(&Grid);
  pRNG.SeedFixedIntegers(seeds);

  ////////////////////////////////////////////////////////////////////////
  // 1. Read the input gauge configuration (or generate a hot one)
  ////////////////////////////////////////////////////////////////////////
  LatticeGaugeField U(&Grid);
  if (!gaugeFile.empty()) {
    readConfig(U, gaugeFile, "gauge");
  } else {
    SU<Nc>::HotConfiguration(pRNG, U);
    std::cout << GridLogMessage
              << "No --gauge given; using hot gauge configuration" << std::endl;
  }

  ////////////////////////////////////////////////////////////////////////
  // 2. HISQ smear the gauge link: fat7 -> U(3) project -> asqtad + Naik.
  //    Xfat = smeared fat links, Wlong = smeared long (Naik) links.
  ////////////////////////////////////////////////////////////////////////
  LatticeGaugeField R(&Grid), V(&Grid), W(&Grid), Xfat(&Grid), Wlong(&Grid);
  {
    HighlyImprovedStaggeredFermionImpl<PeriodicGimplR> hisq(&Grid);

    std::cout << GridLogMessage << "Begin staggered phase multiply"
              << std::endl;
    hisq.rephase(R, U); // phase in (staggered + BC) phases
    std::cout << GridLogMessage << "Begin First smearing pass" << std::endl;
    hisq.smear(V, R); // level-1 fat7 (no Naik)
    std::cout << GridLogMessage << "Begin projection step" << std::endl;
    hisq.project(W, V); // U(3) projection of the fat links
    std::cout << GridLogMessage << "Begin second smearing" << std::endl;
    hisq.smear(Xfat, Wlong, W); // level-2 asqtad: Xfat = fat, Wlong = Naik/long
    std::cout << GridLogMessage << "Smearing complete" << std::endl;
  }

  ////////////////////////////////////////////////////////////////////////
  // 3. Input fat / long links: read (ILDG) or reuse the smeared output
  //    so the test still runs as a smoke test when no files are given.
  ////////////////////////////////////////////////////////////////////////
  LatticeGaugeField Ufat(&Grid), Ulong(&Grid);
  bool filesProvided = (!fatFile.empty() && !longFile.empty());
  if (filesProvided) {
    readConfig(Ufat, fatFile, "fat links");
    readConfig(Ulong, longFile, "long links");
  } else {
    Ufat = Xfat;
    Ulong = Wlong;
    std::cout << GridLogMessage
              << "No --fat/--long given; reusing HISQ-smeared links "
              << "(smoke mode: the residual should be zero)" << std::endl;
  }

  ////////////////////////////////////////////////////////////////////////
  // 4. Residual: || U^input - U^smeared || / || U^smeared ||
  ////////////////////////////////////////////////////////////////////////
  LatticeGaugeField fatDiff(&Grid), longDiff(&Grid);
  fatDiff = Ufat - Xfat;
  longDiff = Ulong - Wlong;

  RealD nFatRef = norm2(Xfat);
  RealD nLongRef = norm2(Wlong);
  RealD resFat = std::sqrt(norm2(fatDiff) / nFatRef);
  RealD resLong = std::sqrt(norm2(longDiff) / nLongRef);

  std::cout
      << GridLogMessage
      << "\n  ---------------------------------------------------------------"
      << "\n  fat-link  residual ||U^in - U^smear|| / ||U^smear||  = " << resFat
      << "\n  long-link residual ||U^in - U^smear|| / ||U^smear||  = "
      << resLong << "\n  (tolerance = " << tol << ")" << std::endl;

  ////////////////////////////////////////////////////////////////////////
  // 5. Checks
  ////////////////////////////////////////////////////////////////////////
  assert(std::isfinite(resFat) && nFatRef > 0.0);
  assert(std::isfinite(resLong) && nLongRef > 0.0);

  int rc = 0;
  if ((resFat < tol) && (resLong < tol)) {
    std::cout << GridLogMessage << "Input fat/long links match HISQ smearing "
              << "within tolerance" << std::endl;
    Grid_pass("fat/long links match HISQ smearing of gauge configuration");
  } else {
    std::cout << GridLogError << "Residual FAILED tolerance" << std::endl;
    Grid_error("input fat/long links disagree with HISQ smearing");
    rc = 1; // fail the test (non-zero exit) on genuine disagreement
  }

  Grid_finalize();
  return rc;
}
