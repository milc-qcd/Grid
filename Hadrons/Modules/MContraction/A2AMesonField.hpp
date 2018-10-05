/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid 

Source file: Hadrons/Modules/MContraction/A2AMesonField.hpp

Copyright (C) 2015-2018

Author: Antonin Portelli <antonin.portelli@me.com>
Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: paboyle <paboyle@ph.ed.ac.uk>

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

See the full license in the file "LICENSE" in the top level distribution directory
*************************************************************************************/
/*  END LEGAL */
#ifndef Hadrons_MContraction_A2AMesonField_hpp_
#define Hadrons_MContraction_A2AMesonField_hpp_

#include <Hadrons/Global.hpp>
#include <Hadrons/Module.hpp>
#include <Hadrons/ModuleFactory.hpp>
#include <Hadrons/A2AVectors.hpp>
#include <Hadrons/A2AMatrix.hpp>
#include <Hadrons/Modules/MSolver/A2AVectors.hpp>
#include <Hadrons/Modules/MContraction/A2AKernels.hpp>

#define MF_PARALLEL_IO
#ifndef MF_IO_TYPE
#define MF_IO_TYPE ComplexF
#endif

BEGIN_HADRONS_NAMESPACE

/******************************************************************************
 *                     All-to-all meson field creation                        *
 ******************************************************************************/
BEGIN_MODULE_NAMESPACE(MContraction)

class A2AMesonFieldPar: Serializable
{
public:
    GRID_SERIALIZABLE_CLASS_MEMBERS(A2AMesonFieldPar,
                                    int, cacheBlock,
                                    int, block,
                                    std::string, v,
                                    std::string, w,
                                    std::string, output,
                                    std::string, gammas,
                                    std::vector<std::string>, mom);
};

class A2AMesonFieldMetadata: Serializable
{
public:
    GRID_SERIALIZABLE_CLASS_MEMBERS(A2AMesonFieldMetadata,
                                    std::vector<RealF>, momentum,
                                    Gamma::Algebra, gamma);
};

template <typename T, typename Field>
class MesonFieldKernel: public A2AKernel<T, Field>
{
public:
    MesonFieldKernel(const std::vector<Gamma::Algebra> &gamma,
                     const std::vector<LatticeComplex> &mom,
                     GridBase *grid)
    : gamma_(gamma), mom_(mom), grid_(grid)
    {
        vol_ = 1.;
        for (auto &d: grid_->GlobalDimensions())
        {
            vol_ *= d;
        }
    }

    virtual ~MesonFieldKernel(void) = default;
    virtual void operator()(A2AMatrixSet<T> &m, const Field *left, const Field *right,
                          const unsigned int orthogDim, double &time)
    {
        makeMesonFieldBlock(m, left, right, gamma_, mom_, orthogDim, time);
    }

    virtual double flops(const unsigned int blockSizei, const unsigned int blockSizej)
    {
        return vol_*(2*8.0+6.0+8.0*mom_.size())*blockSizei*blockSizej*gamma_.size();
    }

    virtual double bytes(const unsigned int blockSizei, const unsigned int blockSizej)
    {
        return vol_*(12.0*sizeof(T))*blockSizei*blockSizej
               +  vol_*(2.0*sizeof(T)*mom_.size())*blockSizei*blockSizej*gamma_.size();
    }
private:
    const std::vector<Gamma::Algebra> &gamma_;
    const std::vector<LatticeComplex> &mom_;
    GridBase                          *grid_;
    double                            vol_;
};

template <typename FImpl>
class TA2AMesonField : public Module<A2AMesonFieldPar>
{
public:
    FERM_TYPE_ALIASES(FImpl,);
    SOLVER_TYPE_ALIASES(FImpl,);
    typedef A2AMatrixBlockComputation<Complex, 
                                      FermionField, 
                                      A2AMesonFieldMetadata, 
                                      MF_IO_TYPE> Computation;
    typedef MesonFieldKernel<Complex, FermionField> Kernel;
    struct IoHelper
    {
        A2AMatrixIo<MF_IO_TYPE> io;
        A2AMesonFieldMetadata   md;
        unsigned int            m, g, i, j;
    };
public:
    // constructor
    TA2AMesonField(const std::string name);
    // destructor
    virtual ~TA2AMesonField(void){};
    // dependency relation
    virtual std::vector<std::string> getInput(void);
    virtual std::vector<std::string> getOutput(void);
    // setup
    virtual void setup(void);
    // execution
    virtual void execute(void);
private:
    // IO
    std::string ioname(const unsigned int m, const unsigned int g) const;
    std::string filename(const unsigned int m, const unsigned int g) const;
    void saveBlock(const A2AMatrixSet<MF_IO_TYPE> &mf, IoHelper &h);
private:
    bool                               hasPhase_{false};
    std::string                        momphName_;
    std::vector<Gamma::Algebra>        gamma_;
    std::vector<std::vector<Real>>     mom_;
    std::vector<IoHelper>              nodeIo_;
};

MODULE_REGISTER(A2AMesonField, ARG(TA2AMesonField<FIMPL>), MContraction);
MODULE_REGISTER(ZA2AMesonField, ARG(TA2AMesonField<ZFIMPL>), MContraction);

/******************************************************************************
*                  TA2AMesonField implementation                             *
******************************************************************************/
// constructor /////////////////////////////////////////////////////////////////
template <typename FImpl>
TA2AMesonField<FImpl>::TA2AMesonField(const std::string name)
: Module<A2AMesonFieldPar>(name)
, momphName_(name + "_momph")
{
}

// dependencies/products ///////////////////////////////////////////////////////
template <typename FImpl>
std::vector<std::string> TA2AMesonField<FImpl>::getInput(void)
{
    std::vector<std::string> in = {par().v, par().w};

    return in;
}

template <typename FImpl>
std::vector<std::string> TA2AMesonField<FImpl>::getOutput(void)
{
    std::vector<std::string> out = {};

    return out;
}

// setup ///////////////////////////////////////////////////////////////////////
template <typename FImpl>
void TA2AMesonField<FImpl>::setup(void)
{
    gamma_.clear();
    mom_.clear();
    if (par().gammas == "all")
    {
        gamma_ = {
            Gamma::Algebra::Gamma5,
            Gamma::Algebra::Identity,    
            Gamma::Algebra::GammaX,
            Gamma::Algebra::GammaY,
            Gamma::Algebra::GammaZ,
            Gamma::Algebra::GammaT,
            Gamma::Algebra::GammaXGamma5,
            Gamma::Algebra::GammaYGamma5,
            Gamma::Algebra::GammaZGamma5,
            Gamma::Algebra::GammaTGamma5,
            Gamma::Algebra::SigmaXY,
            Gamma::Algebra::SigmaXZ,
            Gamma::Algebra::SigmaXT,
            Gamma::Algebra::SigmaYZ,
            Gamma::Algebra::SigmaYT,
            Gamma::Algebra::SigmaZT
        };
    }
    else
    {
        gamma_ = strToVec<Gamma::Algebra>(par().gammas);
    }
    for (auto &pstr: par().mom)
    {
        auto p = strToVec<Real>(pstr);

        if (p.size() != env().getNd() - 1)
        {
            HADRONS_ERROR(Size, "Momentum has " + std::to_string(p.size())
                                + " components instead of " 
                                + std::to_string(env().getNd() - 1));
        }
        mom_.push_back(p);
    }
    
    envCache(std::vector<ComplexField>, momphName_, 1, 
             par().mom.size(), envGetGrid(ComplexField));
    envTmpLat(ComplexField, "coor");
    envTmp(Computation, "computation", 1, envGetGrid(FermionField), 
           env().getNd() - 1, mom_.size(), gamma_.size(), par().block, 
           par().cacheBlock, this);
}

// execution ///////////////////////////////////////////////////////////////////
template <typename FImpl>
void TA2AMesonField<FImpl>::execute(void)
{
    auto &v = envGet(std::vector<FermionField>, par().v);
    auto &w = envGet(std::vector<FermionField>, par().w);

    int nt         = env().getDim().back();
    int N_i        = w.size();
    int N_j        = v.size();
    int ngamma     = gamma_.size();
    int nmom       = mom_.size();
    int block      = par().block;
    int cacheBlock = par().cacheBlock;

    LOG(Message) << "Computing all-to-all meson fields" << std::endl;
    LOG(Message) << "W: '" << par().w << "' V: '" << par().v << "'" << std::endl;
    LOG(Message) << "Momenta:" << std::endl;
    for (auto &p: mom_)
    {
        LOG(Message) << "  " << p << std::endl;
    }
    LOG(Message) << "Spin bilinears:" << std::endl;
    for (auto &g: gamma_)
    {
        LOG(Message) << "  " << g << std::endl;
    }
    LOG(Message) << "Meson field size: " << nt << "*" << N_i << "*" << N_j 
                 << " (filesize " << sizeString(nt*N_i*N_j*sizeof(MF_IO_TYPE)) 
                 << "/momentum/bilinear)" << std::endl;

    ///////////////////////////////////////////////
    // Momentum setup
    ///////////////////////////////////////////////
    auto &ph = envGet(std::vector<ComplexField>, momphName_);

    if (!hasPhase_)
    {
        startTimer("Momentum phases");
        for (unsigned int j = 0; j < nmom; ++j)
        {
            Complex           i(0.0,1.0);
            std::vector<Real> p;

            envGetTmp(ComplexField, coor);
            ph[j] = zero;
            for(unsigned int mu = 0; mu < mom_[j].size(); mu++)
            {
                LatticeCoordinate(coor, mu);
                ph[j] = ph[j] + (mom_[j][mu]/env().getDim(mu))*coor;
            }
            ph[j] = exp((Real)(2*M_PI)*i*ph[j]);
        }
        hasPhase_ = true;
        stopTimer("Momentum phases");
    }

    auto ionameFn = [this](const unsigned int m, const unsigned int g)
    {
        std::stringstream ss;

        ss << gamma_[g] << "_";
        for (unsigned int mu = 0; mu < mom_[m].size(); ++mu)
        {
            ss << mom_[m][mu] << ((mu == mom_[m].size() - 1) ? "" : "_");
        }

        return ss.str();
    };

    auto filenameFn = [this, &ionameFn](const unsigned int m, const unsigned int g)
    {
        return par().output + "." + std::to_string(vm().getTrajectory()) 
               + "/" + ioname(m, g) + ".h5";
    };

    auto metadataFn = [this](const unsigned int m, const unsigned int g)
    {
        A2AMesonFieldMetadata md;

        for (auto pmu: mom_[m])
        {
            md.momentum.push_back(pmu);
        }
        md.gamma = gamma_[g];
        
        return md;
    };

    Kernel      kernel(gamma_, ph, envGetGrid(FermionField));

    envGetTmp(Computation, computation);
    computation.execute(w, v, kernel, ionameFn, filenameFn, metadataFn);
    //////////////////////////////////////////////////////////////////////////
    // i,j   is first  loop over SchurBlock factors reusing 5D matrices
    // ii,jj is second loop over cacheBlock factors for high perf contractoin
    // iii,jjj are loops within cacheBlock
    // Total index is sum of these  i+ii+iii etc...
    //////////////////////////////////////////////////////////////////////////
    
//     double flops;
//     double bytes;
//     double vol      = env().getVolume();
//     double t_kernel = 0.0;
//     double nodes    = env().getGrid()->NodeCount();
//     double tot_kernel;

//     envGetTmp(Vector<MF_IO_TYPE>, mfBuf);
//     envGetTmp(Vector<Complex>, mfCache);
    
//     double t0    = usecond();
//     int NBlock_i = N_i/block + (((N_i % block) != 0) ? 1 : 0);
//     int NBlock_j = N_j/block + (((N_j % block) != 0) ? 1 : 0);

//     for(int i=0;i<N_i;i+=block)
//     for(int j=0;j<N_j;j+=block)
//     {
//         // Get the W and V vectors for this block^2 set of terms
//         int N_ii = MIN(N_i-i,block);
//         int N_jj = MIN(N_j-j,block);

//         LOG(Message) << "Meson field block " 
//                     << j/block + NBlock_j*i/block + 1 
//                     << "/" << NBlock_i*NBlock_j << " [" << i <<" .. " 
//                     << i+N_ii-1 << ", " << j <<" .. " << j+N_jj-1 << "]" 
//                     << std::endl;

//         A2AMatrixSet<MF_IO_TYPE> mfBlock(mfBuf.data(),nmom,ngamma,nt,N_ii,N_jj);

//         // Series of cache blocked chunks of the contractions within this block
//         flops = 0.0;
//         bytes = 0.0;
//         for(int ii=0;ii<N_ii;ii+=cacheBlock)
//         for(int jj=0;jj<N_jj;jj+=cacheBlock)
//         {
//             int N_iii = MIN(N_ii-ii,cacheBlock);
//             int N_jjj = MIN(N_jj-jj,cacheBlock);
//             A2AMatrixSet<Complex> mfCacheBlock(mfCache.data(),nmom,ngamma,nt,N_iii,N_jjj);    

//             startTimer("contraction: total");
//             makeMesonFieldBlock(mfCacheBlock, &w[i+ii], &v[j+jj], gamma_, ph, 
//                                 env().getNd() - 1, this);
//             stopTimer("contraction: total");
            
//             // flops for general N_c & N_s
//             flops += vol * ( 2 * 8.0 + 6.0 + 8.0*nmom) * N_iii*N_jjj*ngamma;
//             bytes += vol * (12.0 * sizeof(Complex) ) * N_iii*N_jjj
//                      +  vol * ( 2.0 * sizeof(Complex) *nmom ) * N_iii*N_jjj* ngamma;

//             startTimer("cache copy");
//             parallel_for_nest5(int m =0;m< nmom;m++)
//             for(int g =0;g< ngamma;g++)
//             for(int t =0;t< nt;t++)
//             for(int iii=0;iii< N_iii;iii++)
//             for(int jjj=0;jjj< N_jjj;jjj++)
//             {
//                 mfBlock(m,g,t,ii+iii,jj+jjj) = mfCacheBlock(m,g,t,iii,jjj);
//             }
//             stopTimer("cache copy");
//         }

//         // perf
//         tot_kernel = getDTimer("contraction: colour trace & mom.")
//                      + getDTimer("contraction: local space sum");
//         t_kernel   = tot_kernel - t_kernel;
//         LOG(Message) << "Kernel perf " << flops/t_kernel/1.0e3/nodes 
//                      << " Gflop/s/node " << std::endl;
//         LOG(Message) << "Kernel perf " << bytes/t_kernel*1.0e6/1024/1024/1024/nodes 
//                      << " GB/s/node "  << std::endl;
//         t_kernel = tot_kernel;

//         // IO
//         if (!par().output.empty())
//         {
//             double       blockSize, ioTime;
//             unsigned int myRank = env().getGrid()->ThisRank(),
//                          nRank  = env().getGrid()->RankCount();
        
//             LOG(Message) << "Writing block to disk" << std::endl;
//             ioTime = -getDTimer("IO: write block");
//             startTimer("IO: total");
//             makeFileDir(filename(0, 0), env().getGrid());
// #ifdef MF_PARALLEL_IO
//             env().getGrid()->Barrier();
//             // make task list for current node
//             nodeIo_.clear();
//             for(int f = myRank; f < nmom*ngamma; f += nRank)
//             {
//                 IoHelper h;

//                 h.i  = i;
//                 h.j  = j;
//                 h.m  = f/ngamma;
//                 h.g  = f % ngamma;
//                 h.io = A2AMatrixIo<MF_IO_TYPE>(filename(h.m, h.g), 
//                                                ioname(h.m, h.g), nt, N_i, N_j);
//                 for (auto pmu: mom_[h.m])
//                 {
//                     h.md.momentum.push_back(pmu);
//                 }
//                 h.md.gamma = gamma_[h.g];
//                 nodeIo_.push_back(h);
//             }
//             // parallel IO
//             for (auto &h: nodeIo_)
//             {
//                 saveBlock(mfBlock, h);
//             }
//             env().getGrid()->Barrier();
// #else
//             // serial IO, for testing purposes only
//             for(int m = 0; m < nmom; m++)
//             for(int g = 0; g < ngamma; g++)
//             {
//                 IoHelper h;

//                 h.i  = i;
//                 h.j  = j;
//                 h.m  = m;
//                 h.g  = g;
//                 h.io = A2AMatrixIo<MF_IO_TYPE>(filename(h.m, h.g), 
//                                                ioname(h.m, h.g), nt, N_i, N_j);
//                 for (auto pmu: mom_[h.m])
//                 {
//                     h.md.momentum.push_back(pmu);
//                 }
//                 h.md.gamma = gamma_[h.g];
//                 saveBlock(mfBlock, h);
//             }
// #endif
//             stopTimer("IO: total");
//             blockSize  = static_cast<double>(nmom*ngamma*nt*N_ii*N_jj*sizeof(MF_IO_TYPE));
//             ioTime    += getDTimer("IO: write block");
//             LOG(Message) << "HDF5 IO done " << sizeString(blockSize) << " in "
//                          << ioTime  << " us (" 
//                          << blockSize/ioTime*1.0e6/1024/1024
//                          << " MB/s)" << std::endl;
//         }
//     }
}

// IO
template <typename FImpl>
std::string TA2AMesonField<FImpl>::ioname(unsigned int m, unsigned int g) const
{
    std::stringstream ss;

    ss << gamma_[g] << "_";
    for (unsigned int mu = 0; mu < mom_[m].size(); ++mu)
    {
        ss << mom_[m][mu] << ((mu == mom_[m].size() - 1) ? "" : "_");
    }

    return ss.str();
}

template <typename FImpl>
std::string TA2AMesonField<FImpl>::filename(unsigned int m, unsigned int g) const
{
    return par().output + "." + std::to_string(vm().getTrajectory()) 
           + "/" + ioname(m, g) + ".h5";
}

template <typename FImpl>
void TA2AMesonField<FImpl>::saveBlock(const A2AMatrixSet<MF_IO_TYPE> &mf, IoHelper &h)
{
    if ((h.i == 0) and (h.j == 0))
    {
        startTimer("IO: file creation");
        h.io.initFile(h.md, par().block);
        stopTimer("IO: file creation");
    }
    startTimer("IO: write block");
    h.io.saveBlock(mf, h.m, h.g, h.i, h.j);
    stopTimer("IO: write block");
}

END_MODULE_NAMESPACE

END_HADRONS_NAMESPACE

#endif // Hadrons_MContraction_A2AMesonField_hpp_