/*************************************************************************************
Grid physics library, www.github.com/paboyle/Grid
Source file: ./lib/qcd/utils/Transporters.h
Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

Copyright (C) 2023 

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

/**
 * @file PeriodicTranspoerters.h
 * @brief Interface for periodic gauge transporters
 * @author Curtis Taylor Peterson
 * @details
 * ...
 */

#pragma once 
#include <Grid/perfmon/Tracing.h>

#ifndef QCD_UTILS_PERIODIC_TRANSPORTERS_H
#define QCD_UTILS_PERIODIC_TRANSPORTERS_H

NAMESPACE_BEGIN(Grid);

namespace PeriodicBC { 

//
// macros
//

// make scope more explicit
#define ACCELERATOR_SCOPE(exec) { exec; } \

// shorten call to get stencil entry in declaration
#define STENCIL_ENTRY(se, st, mu, n)              \
  GeneralStencilEntry const *se = st.GetEntry(mu, n); \

// shorten call to coalesced read in function
#define ACCREAD(u, se)                                          \
  coalescedReadGeneralPermute(u[se->_offset], se->_permute, Nd) \

// shorten call to coalesced write
#define ACCWRITE(wu, u) coalescedWrite(wu, u) \

// shorted gauge loop construction
#define GAUGELOOP1(ua, ub) ua * adj(ub) \

// shorted gauge loop construction
#define GAUGELOOP2(ua, ub) adj(ua) * ub \

// shorten out-of-class implementation definitions
#define IMPL(x) template <class Gimpl>                  \
inline const typename x<Gimpl>::GaugeLinkField x<Gimpl> \

//
// convenient data structures
//

enum TransportHeading {FORWARD = 0, BACKWARD = Nd};

//
// periodic transporter definition
//

template <class Gimpl>
class Transporter: public Gimpl {
/**
 * @brief periodic gauge transporter
 * @author Curtis Taylor Peterson
 * @details
 * Represents periodic transport operations needed for building up common gauge 
 * link constructs. 
 */
public: INHERIT_GIMPL_TYPES(Gimpl)

private:
  int depth, mu;
  std::shared_ptr<GeneralLocalStencil> _stencil;
  std::unique_ptr<GaugeLinkField> _link;

public:
  /** @brief default Transporter constructor */
  Transporter() { _stencil.reset(); _link.reset(); }

  /** @brief main Transporter constructor */
  Transporter(
    GridBase* grid, 
    const GaugeLinkField& link, 
    int mu,
    int depth
  ): mu(mu), depth(depth) {
    GridCartesian* cgrid = (GridCartesian*)grid;
    std::vector<Coordinate> shifts;
    Coordinate backward(Nd, 0);
    Coordinate forward(Nd, 0);
    Coordinate noshift(Nd, 0);

    backward[mu] = -depth;
    forward[mu] = depth;

    shifts.push_back(backward);
    shifts.push_back(forward);
    shifts.push_back(noshift);

    _stencil = std::make_unique<GeneralLocalStencil>(GeneralLocalStencil(cgrid, shifts));
    _link = std::make_unique<GaugeLinkField>(link);
  }

// accessors
public:
  /** @brief return copy of input buffer */
  inline const GaugeLinkField link() { return *_link; }

  /** @brief return copy of general local stencil */
  inline const GeneralLocalStencil stencil() { return *_stencil; }

// covariant shifts
public:
  inline const GaugeLinkField CovShift(
    const GaugeLinkField& u,
    const GaugeLinkField& v, 
    TransportHeading heading
  );

  inline const GaugeLinkField CovShift(
    const GaugeLinkField& v, 
    TransportHeading heading
  );

  inline const GaugeLinkField CovShiftFwd(
    const GaugeLinkField& u, 
    const GaugeLinkField& v
  );

  inline const GaugeLinkField CovShiftFwd(const GaugeLinkField& v);

  inline const GaugeLinkField CovShiftFwd();

  inline const GaugeLinkField CovShiftBck(
    const GaugeLinkField& u, 
    const GaugeLinkField& v
  );

  inline const GaugeLinkField CovShiftBck(const GaugeLinkField& v);

  inline const GaugeLinkField CovShiftBck();

// cartesian shifts
public:
  inline const GaugeLinkField Cshift(
    const GaugeLinkField& v, 
    TransportHeading heading
  );

  inline const GaugeLinkField CovShiftIdent(
    const GaugeLinkField& v, 
    TransportHeading heading
  );

  inline const GaugeLinkField CovShiftIdentFwd(const GaugeLinkField& v);

  inline const GaugeLinkField CovShiftIdentBck(const GaugeLinkField& v);
};

//
// periodic transporter container definition
//

template <class Gimpl>
class Transporters: public Gimpl {
/**
 * @brief Transporter container
 * @author Curtis Taylor Peterson
 * @details
 * Naive container for saving and indexing a collection of Transporters. 
 * Note of caution: all periodic gauge transporters use the same buffer for the 
 * stencil and the outputs.
 */

public: INHERIT_GIMPL_TYPES(Gimpl)

private:
  int depth;

  GridBase* _pgrid;
  PaddedCell* _pcell;
  
  std::shared_ptr<GeneralLocalStencil> _stencil;
  std::vector<std::vector<std::shared_ptr<GeneralLocalStencil>>> _stencils;
  
  Transporter<Gimpl> _transporter[Nd];

public:
  Transporters(PaddedCell& pcell, const GaugeField& Uin): depth(pcell.depth) {
    auto U = pcell.ExchangePeriodic(Uin);
    std::vector<Coordinate> shifts;

    // construct transporters
    
    _pcell = &pcell;
    _pgrid = (GridBase*)_pcell->grids.back();
    for (int mu = 0; mu < Nd; ++mu) 
      _transporter[mu] = Transporter<Gimpl>(_pgrid, toLink(U, mu), mu, depth);

    // construct mu-nu stencils
    
    GridCartesian* cgrid = (GridCartesian*)_pgrid;

    for (int mu = 0; mu < Nd; ++mu) {
      std::vector<std::shared_ptr<GeneralLocalStencil>> stencils;

      for (int nu = 0; nu < Nd; ++nu) {
        if (mu == nu) {
          stencils.push_back(nullptr);
          continue;
        }

        std::vector<Coordinate> shifts;
        Coordinate shift_pmu_pnu(Nd, 0);
        Coordinate shift_pmu_mnu(Nd, 0);
        Coordinate shift_mmu_pnu(Nd, 0);
        Coordinate shift_mmu_mnu(Nd, 0);
        Coordinate shift_0(Nd, 0);

        shift_pmu_pnu[mu] = depth;   shift_pmu_pnu[nu] = depth;
        shift_pmu_mnu[mu] = depth;   shift_pmu_mnu[nu] = -depth;
        shift_mmu_pnu[mu] = -depth;  shift_mmu_pnu[nu] = depth;
        shift_mmu_mnu[mu] = -depth;  shift_mmu_mnu[nu] = -depth;

        shifts.push_back(shift_pmu_pnu);
        shifts.push_back(shift_pmu_mnu);
        shifts.push_back(shift_mmu_pnu);
        shifts.push_back(shift_mmu_mnu);
        shifts.push_back(shift_0);

        stencils.push_back(
          std::make_shared<GeneralLocalStencil>(GeneralLocalStencil(cgrid, shifts))
        );
      }

      _stencils.push_back(stencils);
    }
  }

private:
  GaugeLinkField toLink(const GaugeField& U, int mu)
  { return PeekIndex<LorentzIndex>(U, mu); }

public:
  /** @brief wrapped halo exchange */
  inline const GaugeField toPaddedGrid(const GaugeField& U) 
  { return _pcell->ExchangePeriodic(U); }

  /** @brief wrapped halo exchange */
  inline const GaugeLinkField toPaddedGrid(const GaugeLinkField& U) 
  { return _pcell->ExchangePeriodic(U); }

  /** @brief wrapped halo exchange */
  inline const LatticeComplex toPaddedGrid(const LatticeComplex& U) 
  { return _pcell->ExchangePeriodic(U); }

public:
  /** @brief wrapped extraction from padding */
  inline const GaugeField toTightGrid(const GaugeField& U) 
  { return _pcell->Extract(U); }

  /** @brief wrapped extraction from padding */
  inline const GaugeLinkField toTightGrid(const GaugeLinkField& U) 
  { return _pcell->Extract(U); }

  /** @brief wrapped extraction from padding */
  inline const LatticeComplex toTightGrid(const LatticeComplex& U) 
  { return _pcell->Extract(U); }

public:
  /** 
   * @brief In-place directional halo refresh on the padded grid
   * @details Refreshes halo cells in a single dimension without extracting to the 
   * tight grid and re-padding. Gathers interior face slices, exchanges them via MPI 
   * with neighboring ranks, and scatters the received data into the halo region.
   * For local dimensions (single-process), copies boundary slices into halos.
   * 
   * Modeled after PaddedCell::Face_exchange but operates in-place.
   */
  template<class vobj>
  void refresh(Lattice<vobj>& u, int dim) 
  { _pcell->Face_exchange(u, u, dim, depth, true); }

  /** @brief directional halo exchange -- single dimension */
  inline GaugeLinkField exchange(GaugeLinkField u, int dim) {
    tracePush("Transporters::exchange(dim)");
    refresh(u, dim);
    tracePop("Transporters::exchange(dim)");
    return u;
  }

  /** @brief directional halo exchange -- two dimensions */
  inline GaugeLinkField exchange(GaugeLinkField u, int dim1, int dim2) {
    tracePush("Transporters::exchange(dim1,dim2)");
    refresh(u, dim1);
    refresh(u, dim2);
    tracePop("Transporters::exchange(dim1,dim2)");
    return u;
  }

  /** @brief full halo exchange */
  inline GaugeLinkField exchange(const GaugeLinkField& u) { 
    tracePush("Transporters::exchange");
    GaugeLinkField result = toPaddedGrid(toTightGrid(u)); 
    tracePop("Transporters::exchange");
    return result;
  }

  /** @brief update followed by halo exchange */
  void append(GaugeLinkField& u, const GaugeLinkField& du) 
  { u += du; u = exchange(u); }

public:
  /** @brief index transporters -- mutable */
  inline Transporter<Gimpl>& operator[](int mu) { return _transporter[mu]; }

  /** @brief index transporters -- not mutable */
  inline const Transporter<Gimpl>& operator[](int mu) const 
  { return _transporter[mu]; }

public:
  /** @brief return copy of input buffer for transporter */
  inline const GaugeLinkField link(int mu) { return _transporter[mu].link(); }

  /** @breif return copy of general local stencil from transporters */
  inline const GeneralLocalStencil stencil(int mu) 
  { return _transporter[mu].stencil(); }

  /** @brief return copy of general local stencil for mu-nu pairs */
  inline const GeneralLocalStencil stencil(int mu, int nu) 
  { return *(_stencils[mu][nu]); }

public:
  inline const GaugeLinkField CovShiftFwd(
    int mu, 
    const GaugeLinkField& u, 
    const GaugeLinkField& v
  ) { return _transporter[mu].CovShiftFwd(u, v); }

  inline const GaugeLinkField CovShiftFwd(
    int mu, 
    const GaugeLinkField& v
  ) { return _transporter[mu].CovShiftFwd(v); }

  inline const GaugeLinkField CovShiftFwd(
    int mu, 
    int nu
  ) { return _transporter[mu].CovShiftFwd(link(nu)); }

  inline const GaugeLinkField CovShiftFwd(int mu) 
  { return _transporter[mu].CovShiftFwd(); }

  inline const GaugeLinkField CovShiftBck(
    int mu,
    const GaugeLinkField& u, 
    const GaugeLinkField& v
  ) { return _transporter[mu].CovShiftBck(u, v); }

  inline const GaugeLinkField CovShiftBck(
    int mu, 
    const GaugeLinkField& v
  ) { return _transporter[mu].CovShiftBck(v); }

  inline const GaugeLinkField CovShiftBck(
    int mu, 
    int nu
  ) { return _transporter[mu].CovShiftBck(link(nu)); }

  inline const GaugeLinkField CovShiftBck(int mu) 
  { return _transporter[mu].CovShiftBck(); }

public:
  inline const GaugeLinkField CovShiftIdentFwd(
    int mu, 
    const GaugeLinkField& v
  ) { return _transporter[mu].CovShiftIdentFwd(v); }

  inline const GaugeLinkField CovShiftIdentBck(
    int mu, 
    const GaugeLinkField& v
  ) { return _transporter[mu].CovShiftIdentBck(v); }

  inline const GaugeLinkField CovShiftIdentFwd(
    int mu, 
    int nu
  ) { return _transporter[mu].CovShiftIdentFwd(link(nu)); }

  inline const GaugeLinkField CovShiftIdentBck(
    int mu, 
    int nu
  ) { return _transporter[mu].CovShiftIdentBck(link(nu)); }

public:
  inline const GaugeLinkField reverse(
    const GaugeLinkField& u, 
    int mu
  ) { return CovShiftIdentBck(mu, adj(u)); }

  inline const GaugeLinkField reverse(int mu) 
  { return CovShiftIdentBck(mu, adj(link(mu))); }

public:
  inline const GaugeLinkField _staple(
    const GaugeLinkField& tu,
    const GaugeLinkField& bu, 
    const GaugeLinkField& lv,
    const GaugeLinkField& rv,
    int mu, 
    int nu
  );

public:
  inline const GaugeLinkField staple(
    const GaugeLinkField& tu,
    const GaugeLinkField& bu, 
    const GaugeLinkField& lv,
    const GaugeLinkField& rv,
    int mu, 
    int nu
  );

  inline const GaugeLinkField staple(
    const GaugeLinkField& u,
    const GaugeLinkField& lv,
    const GaugeLinkField& rv,
    int mu, 
    int nu
  );

  inline const GaugeLinkField staple(
    const GaugeLinkField& u, 
    const GaugeLinkField& v,
    int mu, 
    int nu
  );

  inline const GaugeLinkField staple(
    const GaugeLinkField& v, 
    int mu, 
    int nu
  );

  inline const GaugeLinkField staple(int mu, int nu);

public:
  inline const GaugeLinkField _upperStaple(
    const GaugeLinkField& u, 
    const GaugeLinkField& lv,
    const GaugeLinkField& rv,
    int mu, 
    int nu
  );

public:
  inline const GaugeLinkField upperStaple(
    const GaugeLinkField& u, 
    const GaugeLinkField& lv,
    const GaugeLinkField& rv,
    int mu, 
    int nu
  );  

  inline const GaugeLinkField upperStaple(
    const GaugeLinkField& lv, 
    const GaugeLinkField& rv,
    int mu, 
    int nu
  );

  inline const GaugeLinkField upperStaple(
    const GaugeLinkField& u, 
    int mu, 
    int nu
  );

  inline const GaugeLinkField upperStaple(
    int mu, 
    int nu
  );

public:
  inline const GaugeLinkField _lowerStaple(
    const GaugeLinkField& u, 
    const GaugeLinkField& lv,
    const GaugeLinkField& rv,
    int mu, 
    int nu
  );

public:
  inline const GaugeLinkField lowerStaple(
    const GaugeLinkField& u, 
    const GaugeLinkField& lv,
    const GaugeLinkField& rv,
    int mu, 
    int nu
  );

  inline const GaugeLinkField lowerStaple(
    const GaugeLinkField& lv, 
    const GaugeLinkField& rv,
    int mu, 
    int nu
  );

  inline const GaugeLinkField lowerStaple(
    const GaugeLinkField& u, 
    int mu, 
    int nu
  );

  inline const GaugeLinkField lowerStaple(
    int mu, 
    int nu
  );

public:
  inline const GaugeLinkField rightStaple(
    const GaugeLinkField& bv,
    const GaugeLinkField& tv,
    const GaugeLinkField& u,
    int mu, 
    int nu
  );

  inline const GaugeLinkField rightStaple(
    const GaugeLinkField& bv, 
    const GaugeLinkField& tv,
    int mu, 
    int nu
  );

  inline const GaugeLinkField rightStaple(
    const GaugeLinkField& u, 
    int mu, 
    int nu
  );

  inline const GaugeLinkField rightStaple(
    int mu, 
    int nu
  );

public:
  inline const GaugeLinkField leftStaple(
    const GaugeLinkField& bv,
    const GaugeLinkField& tv,
    const GaugeLinkField& u, 
    int mu, 
    int nu
  );

  inline const GaugeLinkField leftStaple(
    const GaugeLinkField& bv, 
    const GaugeLinkField& tv,
    int mu, 
    int nu
  );

  inline const GaugeLinkField leftStaple(
    const GaugeLinkField& u, 
    int mu, 
    int nu
  );

  inline const GaugeLinkField leftStaple(
    int mu, 
    int nu
  );

public:
  inline const GaugeLinkField _stapleDerivative(
    const GaugeLinkField& v,
    const GaugeLinkField& u,
    const GaugeLinkField& c,
    int mu,
    int nu
  );

public:
  inline const GaugeLinkField stapleDerivative(
    const GaugeLinkField& v,
    const GaugeLinkField& u,
    const GaugeLinkField& c,
    int mu,
    int nu
  );

  inline const GaugeLinkField stapleDerivative(
    const GaugeLinkField& v,
    const GaugeLinkField& c,
    int mu,
    int nu
  );

  inline const GaugeLinkField stapleDerivative(
    const GaugeLinkField& c, 
    int mu, 
    int nu
  );
};

//
// periodic transporter implementation
//

//-- covariant shifts --//

/** @brief application of gauge transporter to operand field */
IMPL(Transporter)::CovShift(
  const GaugeLinkField& u,
  const GaugeLinkField& v, 
  TransportHeading heading
) {
  bool forward = heading == FORWARD;
  int direction = forward ? 1 : 0;
  GridBase* pgrid = u.Grid();
  GaugeLinkField f(pgrid);

  tracePush("Transporter::CovShift");
  ACCELERATOR_SCOPE(
    GeneralLocalStencilView s_v = stencil().View(AcceleratorRead);
    autoView(v_v, v, AcceleratorRead);
    autoView(u_v, u, AcceleratorRead);
    autoView(f_v, f, AcceleratorWrite);
    const uint64_t nsimd = pgrid->Nsimd();
    // no overhead from branching inside accelerator loop bc heading constant
    accelerator_for(n, pgrid->oSites(), nsimd, {
      STENCIL_ENTRY(se, s_v, direction, n);
      if (forward) { 
        STENCIL_ENTRY(se_0, s_v, 2, n);
        ACCWRITE(f_v[n], ACCREAD(u_v, se_0)*ACCREAD(v_v, se));
      }
      else ACCWRITE(f_v[n], adj(ACCREAD(u_v, se))*ACCREAD(v_v, se));
    });
  )
  tracePop("Transporter::CovShift");

  return f;
}

/** @brief application of gauge transporter to operand field */
IMPL(Transporter)::CovShift(const GaugeLinkField& v, TransportHeading heading) 
{ return CovShift(link(), v, heading); }

/** @brief application of gauge transporter to operand field */
IMPL(Transporter)::CovShiftFwd(const GaugeLinkField& u, const GaugeLinkField& v) 
{ return CovShift(u, v, FORWARD); }

/** @brief application of gauge transporter to operand field */
IMPL(Transporter)::CovShiftFwd(const GaugeLinkField& v) 
{ return CovShift(link(), v, FORWARD); }

/** @brief application of gauge transporter to operand field */
IMPL(Transporter)::CovShiftFwd() 
{ return CovShift(link(), link(), FORWARD); }

/** @brief application of gauge transporter to operand field */
IMPL(Transporter)::CovShiftBck(const GaugeLinkField& u, const GaugeLinkField& v) 
{ return CovShift(u, v, BACKWARD); }

/** @brief application of gauge transporter to operand field */
IMPL(Transporter)::CovShiftBck(const GaugeLinkField& v) 
{ return CovShift(link(), v, BACKWARD); }

/** @brief application of gauge transporter to operand field */
IMPL(Transporter)::CovShiftBck() 
{ return CovShift(link(), link(), BACKWARD); }

//-- cartesian shifts --//

/** @brief application of shift operation to operand field */
IMPL(Transporter)::Cshift(const GaugeLinkField& u, TransportHeading heading) {
  bool forward = heading == FORWARD;
  int direction = forward ? 1 : 0;
  GridBase* pgrid = u.Grid();
  GaugeLinkField f(pgrid);

  tracePush("Transporter::Cshift");
  ACCELERATOR_SCOPE(
    GeneralLocalStencilView s_v = stencil().View(AcceleratorRead);
    autoView(u_v, u, AcceleratorRead);
    autoView(f_v, f, AcceleratorWrite);
    const uint64_t nsimd = pgrid->Nsimd();

    accelerator_for(n, pgrid->oSites(), nsimd, {
      STENCIL_ENTRY(se, s_v, direction, n);
      ACCWRITE(f_v[n], ACCREAD(u_v, se));
    });
  )
  tracePop("Transporter::Cshift");

  return f;
}

/** @brief slightly more satisfying name for Cartesian shift */
IMPL(Transporter)::CovShiftIdent(const GaugeLinkField& v, TransportHeading heading) 
{ return Cshift(v, heading); }

/** @brief slightly more satisfying name for Cartesian shift */
IMPL(Transporter)::CovShiftIdentFwd(const GaugeLinkField& v) 
{ return Cshift(v, FORWARD); }

/** @brief slightly more satisfying name for Cartesian shift */
IMPL(Transporter)::CovShiftIdentBck(const GaugeLinkField& v) 
{ return Cshift(v, BACKWARD); }

//
// periodic transporters implementation
//

//-- symmetric staple --//

IMPL(Transporters)::_staple(
  const GaugeLinkField& tu, // top mu link
  const GaugeLinkField& bu, // bottom mu link
  const GaugeLinkField& lv, // left nu link
  const GaugeLinkField& rv, // right nu link
  int mu, 
  int nu
) {
  /**
   * @brief Calculates "symmetric" staple (i.e., staple + its reflection)
   * @author Curtis Taylor Peterson
   * @details
   * Calculates closed/symmetric staple with orientation
   *         ---🠢
   *         🠡   🠣
   * ν       x---x
   * 🠡       🠣   🠡
   * -🠢 μ    ---🠢 
   * What is meant by "symmetric" is that the staple and its reflection about
   * the plane bisecting the mu-link are added together.
   */ 
  GaugeLinkField rs(_pgrid);

  tracePush("Transporters::_staple");
  ACCELERATOR_SCOPE(
    GeneralLocalStencilView smu_v = stencil(mu).View(AcceleratorRead);
    GeneralLocalStencilView snu_v = stencil(nu).View(AcceleratorRead);
    GeneralLocalStencilView smunu_v = stencil(mu, nu).View(AcceleratorRead);
    
    autoView(tu_v, tu, AcceleratorRead);
    autoView(bu_v, bu, AcceleratorRead);
    autoView(lv_v, lv, AcceleratorRead);
    autoView(rv_v, rv, AcceleratorRead);

    autoView(rs_v, rs, AcceleratorWrite);
    const uint64_t nsimd = _pgrid->Nsimd();

    accelerator_for(n, _pgrid->oSites(), nsimd, {
      STENCIL_ENTRY(se_pmu, smu_v, 1, n);
      STENCIL_ENTRY(se_mnu, snu_v, 0, n);
      STENCIL_ENTRY(se_pnu, snu_v, 1, n);
      STENCIL_ENTRY(se_pmu_mnu, smunu_v, 1, n);
      STENCIL_ENTRY(se_0, smunu_v, 4, n);

      // upper staple
      auto v_x = ACCREAD(lv_v, se_0);
      auto u_xpnu = ACCREAD(tu_v, se_pnu);
      auto v_xpmu = ACCREAD(rv_v, se_pmu);
      auto staple = v_x*u_xpnu*adj(v_xpmu);

      // lower staple
      auto v_xmnu = ACCREAD(lv_v, se_mnu);
      auto u_xmnu = ACCREAD(bu_v, se_mnu);
      auto v_xmnu_pmu = ACCREAD(rv_v, se_pmu_mnu);
      staple = staple + adj(v_xmnu)*u_xmnu*v_xmnu_pmu;

      // full result
      ACCWRITE(rs_v[n], staple);
    });
  )
  tracePop("Transporters::_staple");

  return rs;
}

IMPL(Transporters)::staple(
  const GaugeLinkField& tu, // top mu link
  const GaugeLinkField& bu, // bottom mu link
  const GaugeLinkField& lv, // left nu link
  const GaugeLinkField& rv, // right nu link
  int mu, 
  int nu
) { return _staple(tu, bu, lv, rv, mu, nu); }

IMPL(Transporters)::staple(
  const GaugeLinkField& u, // mu link
  const GaugeLinkField& lv, // left nu link
  const GaugeLinkField& rv, // right nu link
  int mu, 
  int nu
) { return _staple(u, u, lv, rv, mu, nu); }

IMPL(Transporters)::staple(
  const GaugeLinkField& u, // mu link
  const GaugeLinkField& v, // nu link
  int mu, 
  int nu
) { return _staple(u, u, v, v, mu, nu); }

IMPL(Transporters)::staple(
  const GaugeLinkField& u, 
  int mu, 
  int nu
) { return _staple(u, u, link(nu), link(nu), mu, nu); }

IMPL(Transporters)::staple(int mu, int nu) 
{ return _staple(link(mu), link(mu), link(nu), link(nu), mu, nu); }

// -- upper staple --//

IMPL(Transporters)::_upperStaple(
  const GaugeLinkField& u, // mu link
  const GaugeLinkField& lv, // left nu link
  const GaugeLinkField& rv, // right nu link
  int mu, 
  int nu
) {
  /**
   * @brief Calculates upper staple only
   * @author Curtis Taylor Peterson
   * @details
   * Calculates upper staple only with orientation
   * ν      ---🠢
   * 🠡      🠡   🠣
   * -🠢 μ  x
   */ 
  GaugeLinkField rs(_pgrid);

  tracePush("Transporters::_upperStaple");
  ACCELERATOR_SCOPE(
    GeneralLocalStencilView smu_v = stencil(mu).View(AcceleratorRead);
    GeneralLocalStencilView snu_v = stencil(nu).View(AcceleratorRead);
    GeneralLocalStencilView smunu_v = stencil(mu, nu).View(AcceleratorRead);
    
    autoView(u_v, u, AcceleratorRead);
    autoView(lv_v, lv, AcceleratorRead);
    autoView(rv_v, rv, AcceleratorRead);
    autoView(rs_v, rs, AcceleratorWrite);
    const uint64_t nsimd = _pgrid->Nsimd();

    accelerator_for(n, _pgrid->oSites(), nsimd, {
      STENCIL_ENTRY(se_pmu, smu_v, 1, n);
      STENCIL_ENTRY(se_pnu, snu_v, 1, n);
      STENCIL_ENTRY(se_0, smunu_v, 4, n);

      auto v_x = ACCREAD(lv_v, se_0);
      auto u_xpnu = ACCREAD(u_v, se_pnu);
      auto v_xpmu = ACCREAD(rv_v, se_pmu);
      ACCWRITE(rs_v[n], v_x*u_xpnu*adj(v_xpmu));
    });
  )
  tracePop("Transporters::_upperStaple");

  return rs;
}

IMPL(Transporters)::upperStaple(
  const GaugeLinkField& u, // mu link
  const GaugeLinkField& lv, // left nu link
  const GaugeLinkField& rv, // right nu link
  int mu, 
  int nu
) { return _upperStaple(u, lv, rv, mu, nu); }

IMPL(Transporters)::upperStaple(
  const GaugeLinkField& lv, // mu link
  const GaugeLinkField& rv, // nu link
  int mu, 
  int nu
) { return _upperStaple(link(mu), lv, rv, mu, nu); }

IMPL(Transporters)::upperStaple(
  const GaugeLinkField& u, 
  int mu, 
  int nu
) { return _upperStaple(u, link(nu), link(nu), mu, nu); }

IMPL(Transporters)::upperStaple(int mu, int nu) 
{ return _upperStaple(link(mu), link(nu), link(nu), mu, nu); }

// -- lower staple --//

/** @brief calculate lower staple */
IMPL(Transporters)::_lowerStaple(
  const GaugeLinkField& u, // mu link
  const GaugeLinkField& lv, // left nu link
  const GaugeLinkField& rv, // right nu link
  int mu, 
  int nu
) {
  /**
   * @brief Calculates lower staple only
   * @author Curtis Taylor Peterson
   * @details
   * Calculates lower staple only with orientation
   * ν      x
   * 🠡      🠣   🠡
   * -🠢 μ   ---🠢 
   */ 
  GaugeLinkField rs(_pgrid);

  tracePush("Transporters::_lowerStaple");
  ACCELERATOR_SCOPE(
    GeneralLocalStencilView smu_v = stencil(mu).View(AcceleratorRead);
    GeneralLocalStencilView snu_v = stencil(nu).View(AcceleratorRead);
    GeneralLocalStencilView smunu_v = stencil(mu, nu).View(AcceleratorRead);
    
    autoView(u_v, u, AcceleratorRead);
    autoView(lv_v, lv, AcceleratorRead);
    autoView(rv_v, rv, AcceleratorRead);
    autoView(rs_v, rs, AcceleratorWrite);
    const uint64_t nsimd = _pgrid->Nsimd();

    accelerator_for(n, _pgrid->oSites(), nsimd, {
      STENCIL_ENTRY(se_mnu, snu_v, 0, n);
      STENCIL_ENTRY(se_pmu_mnu, smunu_v, 1, n);

      auto v_xmnu = ACCREAD(lv_v, se_mnu);
      auto u_xmnu = ACCREAD(u_v, se_mnu);
      auto v_xmnu_pmu = ACCREAD(rv_v, se_pmu_mnu);
      ACCWRITE(rs_v[n], adj(v_xmnu)*u_xmnu*v_xmnu_pmu);
    });
  )
  tracePop("Transporters::_lowerStaple");

  return rs;
}

IMPL(Transporters)::lowerStaple(
  const GaugeLinkField& u, // mu link
  const GaugeLinkField& lv, // left nu link
  const GaugeLinkField& rv, // right nu link
  int mu, 
  int nu
) { return _lowerStaple(u, lv, rv, mu, nu); }

IMPL(Transporters)::lowerStaple(
  const GaugeLinkField& lv, // left nu link
  const GaugeLinkField& rv, // right nu link
  int mu, 
  int nu
) { return _lowerStaple(link(mu), lv, rv, mu, nu); }

IMPL(Transporters)::lowerStaple(
  const GaugeLinkField& u, 
  int mu, 
  int nu
) { return _lowerStaple(u, link(nu), link(nu), mu, nu); }

IMPL(Transporters)::lowerStaple(int mu, int nu) 
{ return _lowerStaple(link(mu), link(nu), link(nu), mu, nu); }

//-- right staple --//

/**
 * Right staple orientation:
 *       🠤----      
 * ν          🠡
 * 🠡   x ----🠢      
 * -🠢 μ    
 */ 

IMPL(Transporters)::rightStaple(
  const GaugeLinkField& bv, // bottom mu link
  const GaugeLinkField& tv, // top mu link
  const GaugeLinkField& u, // nu link
  int mu, 
  int nu
) { return _upperStaple(u, bv, tv, nu, mu); }

IMPL(Transporters)::rightStaple(
  const GaugeLinkField& bv, // bottom mu link
  const GaugeLinkField& tv, // top mu link
  int mu, 
  int nu
) { return _upperStaple(link(nu), bv, tv, nu, mu); }

IMPL(Transporters)::rightStaple(
  const GaugeLinkField& u, 
  int mu, 
  int nu
) { return _upperStaple(u, link(mu), link(mu), nu, mu); }

IMPL(Transporters)::rightStaple(int mu, int nu) 
{ return _upperStaple(link(nu), link(mu), link(mu), nu, mu); }

//-- left staple --//

/**
 * Left staple orientation:
 *      ----🠢      
 * ν    🠡
 * 🠡    🠤---- x     
 * -🠢 μ    
 */

IMPL(Transporters)::leftStaple(
  const GaugeLinkField& bv, // bottom mu link
  const GaugeLinkField& tv, // top mu link
  const GaugeLinkField& u, // nu link
  int mu, 
  int nu
) { return _lowerStaple(u, bv, tv, nu, mu); }

IMPL(Transporters)::leftStaple(
  const GaugeLinkField& bv, // bottom mu link
  const GaugeLinkField& tv, // top mu link
  int mu, 
  int nu
) { return _lowerStaple(link(nu), bv, tv, nu, mu); }

IMPL(Transporters)::leftStaple(
  const GaugeLinkField& u, 
  int mu, 
  int nu
) { return _lowerStaple(u, link(mu), link(mu), nu, mu); }

IMPL(Transporters)::leftStaple(int mu, int nu) 
{ return _lowerStaple(link(nu), link(mu), link(mu), nu, mu); }

//-- symmetric staple derivative --//

IMPL(Transporters)::_stapleDerivative(
  const GaugeLinkField& u, // mu link
  const GaugeLinkField& v, // nu link
  const GaugeLinkField& c, // chain
  int mu,
  int nu
) {
  /**
   * @brief Calculates nu-oriented derivative of symmetric staple
   * @author Curtis Taylor Peterson
   * @details
   * Calculates nu-oriented derivative of closed/symmetric staple. See diagram in
   * symmetric staple method for orientation of symmetric staple. Derivative "D"
   * of just nu links (what this method calculates) is
   *            ----🠢      ----🠢 
   *            ⮾    🠣     🠡   ⮾ 
   * ν     D =  x----x  +  x----x
   * 🠡          ⮾    🠡     🠣   ⮾
   * -🠢 μ       ----🠢      ----🠢 
   *             [1a]       [1b] 
   * where replacement of "🠣" or "🠡" with ⮾ indicates replacement of link with 
   * contribution from chain rule (i.e., action of derivative).
   */
  GaugeLinkField rds(_pgrid);

  tracePush("Transporters::_stapleDerivative");
  ACCELERATOR_SCOPE(
    GeneralLocalStencilView smu_v = stencil(mu).View(AcceleratorRead);
    GeneralLocalStencilView snu_v = stencil(nu).View(AcceleratorRead);
    GeneralLocalStencilView smunu_v = stencil(mu, nu).View(AcceleratorRead);
    
    autoView(u_v, u, AcceleratorRead);
    autoView(v_v, v, AcceleratorRead);
    autoView(c_v, c, AcceleratorRead);
    autoView(rds_v, rds, AcceleratorWrite);
    const uint64_t nsimd = _pgrid->Nsimd();

    accelerator_for(n, _pgrid->oSites(), nsimd, {
      STENCIL_ENTRY(se_pmu, smu_v, 1, n);
      STENCIL_ENTRY(se_mnu, snu_v, 0, n);
      STENCIL_ENTRY(se_pnu, snu_v, 1, n);
      STENCIL_ENTRY(se_pmu_mnu, smunu_v, 1, n);
      STENCIL_ENTRY(se_0, smunu_v, 4, n);

      // left: upper staple
      auto c_x = ACCREAD(c_v, se_0);
      auto u_xpnu = ACCREAD(u_v, se_pnu);
      auto v_xpmu = ACCREAD(v_v, se_pmu);
      auto staple = c_x*u_xpnu*adj(v_xpmu);

      // left: lower staple
      auto c_xmnu = ACCREAD(c_v, se_mnu);
      auto u_xmnu = ACCREAD(u_v, se_mnu);
      auto v_xmnu_pmu = ACCREAD(v_v, se_pmu_mnu);
      staple = staple + adj(c_xmnu)*u_xmnu*v_xmnu_pmu;

      // right: upper staple
      auto v_x = ACCREAD(v_v, se_0);
      auto c_xpmu = ACCREAD(c_v, se_pmu);
      staple = staple + v_x*u_xpnu*adj(c_xpmu);

      // right: lower staple
      auto v_xmnu = ACCREAD(v_v, se_mnu);
      auto c_xmnu_pmu = ACCREAD(c_v, se_pmu_mnu);
      staple = staple + adj(v_xmnu)*u_xmnu*c_xmnu_pmu;

      // full result
      ACCWRITE(rds_v[n], staple);
    });
  )
  tracePop("Transporters::_stapleDerivative");
  
  return rds;
}

//-- symmetric staple derivative --//

IMPL(Transporters)::stapleDerivative(
  const GaugeLinkField& u, // mu link
  const GaugeLinkField& v, // nu link
  const GaugeLinkField& c, // chain
  int mu,
  int nu
) { return _stapleDerivative(u, v, c, mu, nu); }

/** @brief symmetric staple derivative w/o passing of middle link */
IMPL(Transporters)::stapleDerivative(
  const GaugeLinkField& v,
  const GaugeLinkField& c,
  int mu,
  int nu
) { return stapleDerivative(link(mu), v, c, mu, nu); }

/** @brief symmetric staple derivative w/o explicit middle/side links */
IMPL(Transporters)::stapleDerivative(
  const GaugeLinkField& c, 
  int mu, 
  int nu
) { return stapleDerivative(link(mu), link(nu), c, mu, nu); }

// undefine macros to prevent conflicts
#undef ACCELERATOR_SCOPE
#undef STENCIL_ENTRY
#undef ACCREAD
#undef ACCWRITE
#undef GAUGELOOP1
#undef GAUGELOOP2
#undef IMPL

} // namespace PeriodicBC

NAMESPACE_END(Grid);

#endif // QCD_UTILS_PERIODIC_TRANSPORTERS_H
