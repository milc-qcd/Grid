/*
 * GridMilc/io/MilcIO.h — part of GridMilc (https://github.com/paboyle/Grid)
 *
 * Self-contained MILC v5 plain gauge-configuration reader. Reads the plain
 * single-precision full-3x3 format into a double-precision GaugeField,
 * detects endianness from the magic number, reimplements the MILC sum29/sum31
 * checksum, and computes the plaquette. Modeled after Grid's NerscIO/OpenQcdIO.
 * Extracted from the HadronsMILC LoadMilc module; the Hadrons Module wrapper
 * is intentionally not included here.
 *
 * GridMilc is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License version 2 (or, at your option,
 * any later version). See COPYING/LICENSE in the top-level distribution.
 */
#ifndef GRIDMILC_MILC_IO_H
#define GRIDMILC_MILC_IO_H

//----------------------------------------------------------------------------
// MILC source provenance
//
// The MILC-origination references (milc_qcd/... file:line) annotated
// throughout this file were verified against the milc_qcd project on its
// 'develop' branch:
//
//   commit 406245f6a2e16e01b9f62fee8b5e7d326652fb9c  (406245f6)
//   "Update to accommodate multimass light / single heavy split"
//   2026-07-02
//
//  If you re-validate the references against a later commit, update this block.
//----------------------------------------------------------------------------

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <Grid/Grid.h>

NAMESPACE_BEGIN(Grid)

//----------------------------------------------------------------------------
// MILC v5 on-disk constants — provenance:
//   milc_qcd/include/file_types.h:22   GAUGE_VERSION_NUMBER 0x4e87 (versions 5-7)
//   milc_qcd/generic/io_lat4.c:42      NATURAL_ORDER 0
//   milc_qcd/include/io_lat.h:91       MAX_TIME_STAMP 64
//----------------------------------------------------------------------------
static constexpr uint32_t MILC_V5_MAGIC        = 0x4e87u; // file_types.h:22 GAUGE_VERSION_NUMBER
static constexpr int      MILC_NATURAL_ORDER   = 0;       // io_lat4.c:42 natural/serial site order
static constexpr int      MILC_HEADER_BYTES    = 88;      // magic(4)+dims(16)+time_stamp(64)+order(4)
static constexpr int      MILC_CHECKSUM_BYTES  = 8;       // sum29 (4) + sum31 (4)
static constexpr int      MILC_PAYLOAD_OFFSET  = MILC_HEADER_BYTES + MILC_CHECKSUM_BYTES; // 96

//----------------------------------------------------------------------------
// Container for the parsed MILC v5 binary header + stored checksum block.
// Mirrors the MILC on-disk layout — provenance:
//   milc_qcd/include/io_lat.h:88-103   gauge_header (in-memory field order is
//                                      magic_number, time_stamp[64], dims[4],
//                                      header_bytes, order — NOT the on-disk
//                                      order; see swrite_gauge_hdr below)
//   milc_qcd/include/io_lat.h:126-130  gauge_check (in-memory field order is
//                                      sum31 then sum29)
//   milc_qcd/generic/io_lat_utils.c:395-414  swrite_gauge_hdr serializes the
//                                      header as magic(4), dims[4](16),
//                                      time_stamp[64], order(4) = 88 B,
//                                      followed by the checksum block
//                                      sum29(4)@88, sum31(4)@92 (note:
//                                      written in the OPPOSITE order to the
//                                      in-memory gauge_check struct fields).
//----------------------------------------------------------------------------
struct MilcHeader
{
    uint32_t         magic{0};
    std::vector<int> dims{std::vector<int>(4, 0)};
    std::string      time_stamp;
    int              order{0};
    uint32_t         sum29{0};
    uint32_t         sum31{0};
    bool             byteReverse{false}; // MILC byterevflag: host must swap bytes
};

//----------------------------------------------------------------------------
// Self-contained MILC v5 reader. Reads the plain single-precision full-3x3
// format into a double-precision GaugeField, detects endianness from the
// magic number, reimplements the MILC sum29/sum31 checksum, and computes the
// plaquette. Modeled after Grid's NerscIO/OpenQcdIO.
//----------------------------------------------------------------------------
class MilcIO : public BinaryIO {
public:
    typedef Lattice<vLorentzColourMatrixD> GaugeField;

    //----- helper: is the host big-endian? ---------------------------------
    static inline bool hostIsBigEndian(void)
    {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
        return true;
#elif defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
        return false;
#else
        uint32_t i = 0x01020304u;
        char     c;
        std::memcpy(&c, &i, sizeof(char));
        return (c == 0x01);
#endif
    }

    //----- helper: Grid format string for the detected file endianness -----
    static inline std::string milcFileFormat(const bool byteReverse)
    {
        const bool fileBigEndian = byteReverse ? !hostIsBigEndian()
                                               : hostIsBigEndian();
        return fileBigEndian ? std::string("IEEE32BIG") : std::string("IEEE32");
    }

    //----- read + parse the 96-byte header/checksum prefix -----------------
    static inline int readHeader(std::string file, GridBase *grid,
                                 MilcHeader &header)
    {
        std::ifstream fin(file, std::ios::in | std::ios::binary);
        if (!fin.is_open())
        {
            std::cout << GridLogError << "cannot open MILC gauge file '" << file << "'"
                         << std::endl;
            GRID_ASSERT(0);
        }

        //--- magic number (4 bytes) + endianness detection ---
        // Provenance: milc_qcd/generic/io_lat_utils.c:1280,1336-1351 (read_gauge_hdr)
        //   tmp  == GAUGE_VERSION_NUMBER => byterevflag = 0 (host order)
        //   btmp == GAUGE_VERSION_NUMBER => byterevflag = 1 (byte-reversed),
        //   where btmp = byterevn(magic). Grid::byte_reverse32 corresponds to
        //   milc_qcd/libraries/byterevn.c:11 (byterevn).
        uint32_t rawMagic = 0;
        fin.read(reinterpret_cast<char *>(&rawMagic), sizeof(uint32_t));
        GRID_ASSERT(!fin.fail());
        bool byteReverse;
        if (rawMagic == MILC_V5_MAGIC)
        {
            byteReverse = false;
        }
        else if (byte_reverse32(rawMagic) == MILC_V5_MAGIC)
        {
            byteReverse = true;
        }
        else
        {
            std::cout << GridLogError << "unrecognized magic number in MILC gauge file '"
                         << file << "' (expected 0x4e87); not a plain MILC v5 "
                         << "configuration (archive/SciDAC/old formats are "
                         << "unsupported)" << std::endl;
            GRID_ASSERT(0);
        }
        header.magic       = MILC_V5_MAGIC;
        header.byteReverse = byteReverse;

        //--- dims[4] (16 B) ---
        uint32_t dims[4];
        fin.read(reinterpret_cast<char *>(dims), sizeof(dims));
        GRID_ASSERT(!fin.fail());

        //--- time_stamp[64] ---
        char ts[64];
        fin.read(ts, sizeof(ts));
        GRID_ASSERT(!fin.fail());

        //--- order (4 B) ---
        uint32_t order = 0;
        fin.read(reinterpret_cast<char *>(&order), sizeof(order));
        GRID_ASSERT(!fin.fail());

        // byte-swap the integer header fields to host order if needed
        if (byteReverse)
        {
            for (int d = 0; d < 4; d++)
                dims[d] = byte_reverse32(dims[d]);
            order = byte_reverse32(order);
        }
        header.dims.assign(dims, dims + 4);
        header.time_stamp = std::string(ts, sizeof(ts));
        {
            const auto nul = header.time_stamp.find('\0');
            if (nul != std::string::npos)
                header.time_stamp.resize(nul);
        }
        header.order = static_cast<int>(order);

        //--- checksum block: sum29@88, sum31@92 (note: sum29 first on disk) ---
        uint32_t sum29 = 0, sum31 = 0;
        fin.read(reinterpret_cast<char *>(&sum29), sizeof(sum29));
        fin.read(reinterpret_cast<char *>(&sum31), sizeof(sum31));
        GRID_ASSERT(!fin.fail());
        if (byteReverse)
        {
            sum29 = byte_reverse32(sum29);
            sum31 = byte_reverse32(sum31);
        }
        header.sum29 = sum29;
        header.sum31 = sum31;

        const int data_start = static_cast<int>(fin.tellg()); // 96
        fin.close();

        //--- validate site ordering (coordinate-list/checkpoint unsupported) ---
        if (header.order != MILC_NATURAL_ORDER)
        {
            std::cout << GridLogError << "unsupported site order "
                         << std::to_string(header.order)
                         << " in MILC gauge file '" << file
                         << "' (only natural order / 0 is supported)"
                         << std::endl;
            GRID_ASSERT(0);
        }

        //--- validate lattice dimensions against the grid ---
        GRID_ASSERT(grid->_ndimension == Nd);
        for (int d = 0; d < Nd; d++)
            GRID_ASSERT(static_cast<int>(grid->_fdimensions[d]) == header.dims[d]);

        return data_start;
    }

    //----- MILC sum29/sum31 checksum (local contribution) -----------------
    // Provenance: milc_qcd/generic/io_lat4.c:167-181 (accum_cksums) and the
    //   checksum design rationale at io_lat4.c:49-67. MILC treats each 32-bit
    //   float as an unsigned integer v(i); sum29 left-rotates v by (i mod 29)
    //   bits and XORs the accumulator, sum31 by (i mod 31). The rotate is
    //   written (v<<rank) | (rank==0 ? 0 : v>>(32-rank)) to avoid 32-bit UB on
    //   a zero shift. Here the GLOBAL word index i = global_site*size32 + k
    //   replaces MILC's per-rank counter (rank29/rank31), and per-thread local
    //   accumulators are XOR-reduced by the caller (GlobalXOR) — matching
    //   MILC's commutative/associative design that permits parallel checksum
    //   accumulation (see io_lat4.c:49-67).
    template <class fobj>
    static inline void milcChecksum(GridBase *grid, std::vector<fobj> &fbuf,
                                    uint32_t &milc_csum29, uint32_t &milc_csum31)
    {
        const uint64_t size32 = sizeof(fobj) / sizeof(uint32_t); // 72 for fsu3_matrix[4]
        const int      nd     = grid->_ndimension;
        const uint64_t lsites = grid->lSites();

        Coordinate local_vol   = grid->LocalDimensions();
        Coordinate local_start = grid->LocalStarts();
        Coordinate global_vol  = grid->FullDimensions();

        milc_csum29 = 0;
        milc_csum31 = 0;

        thread_region
        {
            Coordinate coor(nd);
            uint32_t   csum29_thr = 0, csum31_thr = 0;

            thread_for_in_region(local_site, lsites,
            {
                // local lexicographic site -> global site (x-fastest, == MILC natural order)
                Lexicographic::CoorFromIndex(coor, local_site, local_vol);
                for (int d = 0; d < nd; d++)
                    coor[d] += local_start[d];
                int64_t global_site;
                Lexicographic::IndexFromCoor(coor, global_site, global_vol);

                const uint32_t *site_buf
                    = reinterpret_cast<const uint32_t *>(&fbuf[local_site]);
                const uint64_t  base
                    = static_cast<uint64_t>(global_site) * size32;

                for (uint64_t k = 0; k < size32; k++)
                {
                    const uint64_t  idx    = base + k;
                    const uint32_t  v      = site_buf[k];
                    const uint32_t  rank29 = idx % 29u;
                    const uint32_t  rank31 = idx % 31u;

                    csum29_thr ^= (v << rank29)
                                | (rank29 == 0u ? 0u : v >> (32u - rank29));
                    csum31_thr ^= (v << rank31)
                                | (rank31 == 0u ? 0u : v >> (32u - rank31));
                }
            });

            thread_critical
            {
                milc_csum29 ^= csum29_thr;
                milc_csum31 ^= csum31_thr;
            }
        }
    }

    //----- read a MILC v5 configuration into a double GaugeField ----------
    template <class GaugeStats = PeriodicGaugeStatistics>
    static inline void readConfiguration(GaugeField &Umu, MilcHeader &header,
                                         std::string file,
                                         const bool  exitOnMismatch = false,
                                         GaugeStats  GaugeStatisticsCalculator
                                             = GaugeStats())
    {
        GridBase *grid = Umu.Grid();

        uint64_t       offset = readHeader(file, grid, header);
        const std::string format = milcFileFormat(header.byteReverse);

        std::cout << GridLogMessage << "MILC configuration '" << file << "' "
                     << (header.byteReverse ? "byte-swapped" : "host-order")
                     << " (" << format << "), time stamp '"
                     << header.time_stamp << "', site order " << header.order
                     << ", lattice " << header.dims[0] << "x" << header.dims[1]
                     << "x" << header.dims[2] << "x" << header.dims[3]
                     << std::endl;

        typedef LorentzColourMatrixF fobj; // on-disk single-precision fsu3_matrix[4]
        typedef typename GaugeField::vector_object::scalar_object sobj; // double

        const uint64_t       lsites = grid->lSites();
        std::vector<fobj>    iodata(lsites);     // endian-corrected on disk objects
        std::vector<sobj>    scalardata(lsites); // munged double objects

        // parallel MPI-IO read; IOobject swaps iodata to host order in place
        // (and computes Grid's own Nersc/SciDAC checksums, unused here).
        float    wf = 0.f;
        uint32_t nersc_csum, scidac_csuma, scidac_csumb;
        IOobject(wf, grid, iodata, file, offset, format,
                 BINARYIO_READ | BINARYIO_LEXICOGRAPHIC,
                 nersc_csum, scidac_csuma, scidac_csumb);

        // MILC sum29/sum31 over the endian-corrected payload.
        uint32_t milc_csum29, milc_csum31;
        milcChecksum(grid, iodata, milc_csum29, milc_csum31);
        grid->GlobalXOR(milc_csum29);
        grid->GlobalXOR(milc_csum31);

        const bool match29 = (milc_csum29 == header.sum29);
        const bool match31 = (milc_csum31 == header.sum31);

        if (grid->IsBoss())
        {
            std::cout << GridLogMessage << "MILC checksum '" << file << "': "
                         << "sum29 computed 0x" << std::hex << milc_csum29
                         << " stored 0x" << header.sum29
                         << (match29 ? " OK" : " MISMATCH")
                         << " | sum31 computed 0x" << milc_csum31
                         << " stored 0x" << header.sum31
                         << (match31 ? " OK" : " MISMATCH")
                         << std::dec << std::endl;
        }
        if (!match29 || !match31)
        {
            if (exitOnMismatch)
            {
                std::cout << GridLogError << "MILC gauge configuration checksum mismatch "
                             << "in '" << file << "'" << std::endl;
                GRID_ASSERT(0);
            }
            else if (grid->IsBoss())
            {
                std::cout << GridLogMessage << "MILC checksum mismatch in '" << file
                             << "' (continuing; set exitOnChecksumMismatch=true "
                             << "to make fatal)" << std::endl;
            }
        }

        // munge float -> double (element-wise, no transpose/conjugate) + vectorize
        GaugeSimpleMunger<fobj, sobj> munge;
        thread_for(x, lsites, { munge(iodata[x], scalardata[x]); });
        vectorizeFromLexOrdArray(scalardata, Umu);
        grid->Barrier();

        // plaquette / link trace (informational; MILC binary header carries none)
        FieldMetaData stats;
        GaugeStatisticsCalculator(Umu, stats);
        std::cout << GridLogMessage << "MILC configuration '" << file << "' plaquette "
                     << stats.plaquette << " link_trace " << stats.link_trace
                     << std::endl;
    }
};

NAMESPACE_END(Grid)

#endif // GRIDMILC_MILC_IO_H
