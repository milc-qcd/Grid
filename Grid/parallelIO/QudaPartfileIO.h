/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/parallelIO/QudaPartfileIO.h

Copyright (C) 2015

    Author: Carleton DeTar <detar@physics.utah.edu>
    Author: ClaudeCode Sonnet 4.6 9/13/2026

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
#ifndef GRID_QUDA_PARTFILE_IO_H
#define GRID_QUDA_PARTFILE_IO_H

#ifdef HAVE_LIME

#include <Grid/parallelIO/IldgIO.h>

NAMESPACE_BEGIN(Grid);

///////////////////////////////////////////////////////////////////////////////
// Extra XML record types used in QUDA eigenvector partfiles
///////////////////////////////////////////////////////////////////////////////

// Carried in scidac-file-xml of the file-header message.
struct qudaEigenpackFileXml : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(qudaEigenpackFileXml,
                                  int,         Nvec,
                                  std::string, precision,
                                  std::string, version);
  qudaEigenpackFileXml() : Nvec(0), precision("F"), version("1.0") {}
};

// Carried in scidac-record-xml of the global eigenvalue record.
struct qudaEigenvalueRecordXml : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(qudaEigenvalueRecordXml, int, Nvec);
  qudaEigenvalueRecordXml() : Nvec(0) {}
};

// Carried in scidac-record-xml of the eigenvector field record.
struct qudaEvecRecordXml : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(qudaEvecRecordXml,
                                  int,         nColor,
                                  int,         nSpin,
                                  int,         Nvec,
                                  std::string, subset,
                                  std::string, parity,
                                  std::string, precision);
  qudaEvecRecordXml() : nColor(3), nSpin(1), Nvec(0),
                        subset("full"), parity("full"), precision("F") {}
};

///////////////////////////////////////////////////////////////////////////////
// Utilities
///////////////////////////////////////////////////////////////////////////////

// Partfile name for a given MPI rank: <stem>.vol<NNNN>
// QMP node number of this rank's processor coordinate: lexicographic with x
// fastest (QMP's get_rank/lex_rank).  QIO opens <stem>.vol<node> on QMP node
// <node>, whose sublattice is fixed by this coordinate.  Grid's MPI rank is
// not used: MPI_Cart_create orders ranks with the last dimension fastest,
// and Grid may also reorder world ranks for shared memory.
static inline int qudaPartfileRank(GridBase *grid)
{
  const Coordinate &pcoor = grid->ThisProcessorCoor();
  const Coordinate &pgrid = grid->ProcessorGrid();
  int rank = 0;
  for (int d = grid->_ndimension - 1; d >= 0; d--) rank = rank * pgrid[d] + pcoor[d];
  return rank;
}

// Grid rank of QIO's master I/O node (QMP node 0 = processor coordinate 0).
static inline int qudaMasterRank(GridBase *grid)
{
  Coordinate zero(grid->_ndimension, 0);
  return grid->RankFromProcessorCoor(zero);
}

static inline std::string qudaPartfileName(const std::string &stem, int rank)
{
  char buf[8];
  snprintf(buf, sizeof(buf), "%04d", rank);
  return stem + ".vol" + buf;
}

// Text of the first <tag>...</tag> element in xml, or "" if absent.
// Used for record XML whose root element differs between writers:
// Grid writes <qudaEvecRecordXml>, QUDA writes <qudaStaggeredColorSpinorField>
// (or Wilson/MG variants) with subset and parity inside <info>.
static inline std::string qudaXmlTag(const std::string &xml, const std::string &tag)
{
  std::string open = "<" + tag + ">", close = "</" + tag + ">";
  size_t b = xml.find(open);
  if (b == std::string::npos) return "";
  b += open.size();
  size_t e = xml.find(close, b);
  if (e == std::string::npos) return "";
  return xml.substr(b, e - b);
}

// Accumulate one site's contribution to the SciDAC/DML checksum.
// buf must be in big-endian byte order (as stored in the file).
static inline void qudaAccumChecksum(uint64_t gsite, const void *buf,
                                     size_t nbytes,
                                     uint32_t &suma, uint32_t &sumb)
{
  // Rotate-left as in QIO's DML_rank_rotl32.  The naive
  // (c << r) | (c >> (32 - r)) shifts by 32 when r == 0, which is undefined
  // behaviour; with a constant site index (the global eigenvalue record,
  // gsite = 0) the compiler folded it to garbage.
  auto rotl32 = [](uint32_t c, uint32_t r) -> uint32_t {
    r &= 31;
    return (c << r) | (c >> ((32 - r) & 31));
  };
  uint32_t c = crc32(0, (const unsigned char *)buf, nbytes);
  suma ^= rotl32(c, (uint32_t)(gsite % 29));
  sumb ^= rotl32(c, (uint32_t)(gsite % 31));
}

// Site geometry of the I/O lattice.
//
// Full grid:      the I/O lattice is the full lattice.
// Red-black grid: the I/O lattice is QUDA's single-parity lattice
//                 (X/2, Y, Z, T) -- what QUDA's layout_hyper uses for
//                 QUDA_PARITY_SITE_SUBSET.  Sites are lexicographic with
//                 x_cb = x/2 fastest, and the global site index used in the
//                 sitelist and checksums is lexicographic on (X/2, Y, Z, T).
//
// For a red-black grid Grid's LocalDimensions/GlobalDimensions/LocalStarts
// are already checkerboarded in x, so they are exactly the I/O lattice.
// (FullDimensions would give the uncheckerboarded X, which QUDA rejects.)
//
// qudaIoSite(): for local I/O site index s in [0, lSites()), return the
// global I/O site index and set lfull to the full local coordinate expected
// by peekLocalSite/pokeLocalSite.
static inline int64_t qudaIoSite(GridBase *grid, int cb, int s, Coordinate &lfull)
{
  int nd = grid->_ndimension;
  const Coordinate &io_ldims  = grid->LocalDimensions();
  const Coordinate &io_gdims  = grid->GlobalDimensions();
  const Coordinate &io_lstart = grid->LocalStarts();

  Coordinate lcoor(nd), gcoor(nd);
  Lexicographic::CoorFromIndex(lcoor, s, io_ldims);
  for (int d = 0; d < nd; d++) gcoor[d] = lcoor[d] + io_lstart[d];
  int64_t gs;
  Lexicographic::IndexFromCoor(gcoor, gs, io_gdims);

  lfull = lcoor;
  if (grid->_isCheckerBoarded) {
    GRID_ASSERT(grid->CheckerBoarded(0));   // QUDA checkerboards in x
    // Choose the low bit of x so that the global coordinate sum has parity cb.
    int psum = cb;
    for (int d = 1; d < nd; d++) psum += gcoor[d];
    lfull[0] = 2 * lcoor[0] + (psum & 1);
  }
  return gs;
}

///////////////////////////////////////////////////////////////////////////////
// QudaPartfileWriter
//
// Every MPI rank writes its own partfile.  Extends GridLimeWriter with
// boss_node always true.  Follows QIO's PARTFILE convention: only the master
// node's partfile (QMP node 0, vol0000) carries the XML records, checksums
// and global records.  Every other partfile holds just
//
//   Message 1:  scidac-sitelist    (MB=1, ME=1)
//   Message 2:  scidac-binary-data (MB=1, ME=1)  this node's eigenvectors
//
// The checksum in vol0000 is the XOR over all nodes (QIO's
// DML_checksum_combine).  The field record calls a collective (GlobalXOR),
// so all ranks must call writeEvecFieldRecord.
//
// Master partfile layout (three LIME messages):
//
//   Message 1:  scidac-private-file-xml (MB=1, ME=0)
//               scidac-sitelist         (MB=0, ME=0)
//               scidac-file-xml         (MB=0, ME=1)
//
//   Message 2:  scidac-private-record-xml (MB=1, ME=0)  recordtype=GRID_IO_FIELD
//               scidac-record-xml         (MB=0, ME=0)
//               scidac-binary-data        (MB=0, ME=0)  eigenvectors
//               scidac-checksum           (MB=0, ME=1)
//
//   Message 3:  scidac-private-record-xml (MB=1, ME=0)  recordtype=GRID_IO_GLOBAL
//               scidac-record-xml         (MB=0, ME=0)
//               scidac-binary-data        (MB=0, ME=0)  eigenvalues
//               scidac-checksum           (MB=0, ME=1)
//
// The field record must come first: QUDA's read_spinor_field reads only the
// first record and requires datacount == Nvec.  The trailing eigenvalue
// record is ignored by QUDA (it recomputes eigenvalues).
//
// For red-black (single-parity) fields the file describes QUDA's half
// lattice (X/2, Y, Z, T); see qudaIoSite().
//
// On-disk eigenvector layout (big-endian):
//   For each local I/O site s, for each vector i in [0, Nvec):
//     [s0c0.re, s0c0.im, s0c1.re, ..., s(nSpin-1)c(nColor-1).im]
//
// Grid's sobj layout (spin-major, color-minor, re-im) is identical to this,
// so we cast sobj directly to a flat float/double array.
// Extra memory: chunk_sites * Nvec * vlen * elem_bytes (~4.8 MB default).
///////////////////////////////////////////////////////////////////////////////
class QudaPartfileWriter : public GridLimeWriter {
public:
  QudaPartfileWriter() : GridLimeWriter(/*isboss=*/true) {}

  // Open <stem>.vol<NNNN>, NNNN = QMP node number of this rank (qudaPartfileRank).
  void open(const std::string &stem, GridBase *grid)
  {
    int node = qudaPartfileRank(grid);
    master   = (node == 0);
    GridLimeWriter::open(qudaPartfileName(stem, node));
  }

  // -------------------------------------------------------------------------
  // Message 1: file header
  // -------------------------------------------------------------------------
  void writeFileHeader(GridBase *grid, int Nvec,
                       const std::string &precision = "F")
  {
    if (!master) {
      // Non-master partfile: the header is the sitelist alone.
      writeSitelistRecord(grid, 1, 1);
      return;
    }

    // scidac-private-file-xml (MB=1, ME=0)
    scidacFile sf(grid);
    sf.setDimensions(grid->GlobalDimensions());   // X/2 for red-black grids
    sf.volfmt = GRID_IO_MULTIFILE;
    writeLimeObject(1, 0, sf, std::string("scidacFile"), SCIDAC_PRIVATE_FILE_XML);

    // scidac-sitelist (MB=0, ME=0): lsites BE uint64_t global site indices
    writeSitelistRecord(grid, 0, 0);

    // scidac-file-xml (MB=0, ME=1)
    qudaEigenpackFileXml fxml;
    fxml.Nvec      = Nvec;
    fxml.precision = precision;
    fxml.version   = "1.0";
    writeLimeObject(0, 1, fxml, std::string("qudaEigenpackFileXml"), SCIDAC_FILE_XML);
  }

  // -------------------------------------------------------------------------
  // Message 2: global eigenvalue record
  //
  // evals: Nvec complex eigenvalues written as Nvec*2 big-endian doubles.
  // SciDAC checksum uses gsite=0 for global records (single logical "site").
  // -------------------------------------------------------------------------
  void writeGlobalEvalRecord(const std::vector<ComplexD> &evals)
  {
    if (!master) return;                        // global data: master only

    int Nvec = (int)evals.size();

    // scidac-private-record-xml (MB=1, ME=0)
    scidacRecord prec;
    prec.version    = 1.0;
    prec.recordtype = GRID_IO_GLOBAL;
    prec.datatype   = "QUDA_EigenValues";
    prec.precision  = "D";
    prec.colors     = 0;
    prec.spins      = 0;
    prec.typesize   = Nvec * (int)sizeof(double) * 2;
    prec.datacount  = 1;
    writeLimeObject(1, 0, prec, std::string("scidacRecord"), SCIDAC_PRIVATE_RECORD_XML);

    // scidac-record-xml (MB=0, ME=0)
    qudaEigenvalueRecordXml rxml;
    rxml.Nvec = Nvec;
    writeLimeObject(0, 0, rxml, std::string("qudaEigenvalueRecordXml"), SCIDAC_RECORD_XML);

    // Pack: [re0, im0, re1, im1, ...]
    std::vector<double> buf(Nvec * 2);
    for (int i = 0; i < Nvec; i++) {
      buf[2*i]   = evals[i].real();
      buf[2*i+1] = evals[i].imag();
    }

    // Byte-swap to BE; checksum computed on BE data.
    uint64_t nbytes = (uint64_t)buf.size() * sizeof(double);
    htobe64_v(buf.data(), (uint32_t)nbytes);

    uint32_t suma = 0, sumb = 0;
    qudaAccumChecksum(0, buf.data(), nbytes, suma, sumb);

    // scidac-binary-data (MB=0, ME=0)
    writeLimeBinaryRecord(buf.data(), nbytes, 0, 0, SCIDAC_BINARY_DATA);

    // scidac-checksum (MB=0, ME=1)
    writeChecksumRecord(suma, sumb, 0, 1);
  }

  // -------------------------------------------------------------------------
  // Message 3: eigenvector field record
  //
  // Writes site-major, vector-major data: for each site, for each of Nvec
  // vectors, vlen = nSpin*nColor*2 real values in spin-major, color-minor,
  // re-im order (big-endian).
  //
  // Memory overhead: ~chunk_sites * Nvec * vlen * elem_bytes (~4.8 MB default).
  // -------------------------------------------------------------------------
  template <class vobj>
  void writeEvecFieldRecord(std::vector<Lattice<vobj>> &evecs,
                            GridBase *grid,
                            int nColor, int nSpin,
                            bool singlePrec = true,
                            int chunk_sites = 1024)
  {
    typedef typename vobj::scalar_object sobj;
    typedef typename sobj::scalar_type   SC;
    typedef typename SC::value_type     SR;   // float or double (Grid native)

    int Nvec   = (int)evecs.size();
    int nd     = grid->_ndimension;
    int vlen   = nSpin * nColor * 2;            // real values per (site, vec)
    int lsites = grid->lSites();                // half volume on red-black grid
    int cb     = evecs[0].Checkerboard();
    for (int i = 0; i < Nvec; i++) GRID_ASSERT(evecs[i].Checkerboard() == cb);

    // Grid's sobj layout is spin-major, color-minor, re-im, same as QUDA.
    assert((int)(sizeof(sobj) / sizeof(SR)) == vlen);

    int elem_bytes = singlePrec ? (int)sizeof(float) : (int)sizeof(double);

    if (master) {
    // scidac-private-record-xml (MB=1, ME=0)
    scidacRecord prec;
    prec.version    = 1.0;
    prec.recordtype = GRID_IO_FIELD;
    prec.datatype   = "QUDA_ColorSpinorField";
    prec.precision  = singlePrec ? "F" : "D";
    prec.colors     = nColor;
    prec.spins      = nSpin;
    prec.typesize   = vlen * elem_bytes;
    prec.datacount  = Nvec;
    writeLimeObject(1, 0, prec, std::string("scidacRecord"), SCIDAC_PRIVATE_RECORD_XML);

    // scidac-record-xml (MB=0, ME=0)
    qudaEvecRecordXml rxml;
    rxml.nColor    = nColor;
    rxml.nSpin     = nSpin;
    rxml.Nvec      = Nvec;
    if (grid->_isCheckerBoarded) {
      rxml.subset = "parity";
      rxml.parity = (cb == Even) ? "even" : "odd";
    } else {
      rxml.subset = "full";
      rxml.parity = "full";
    }
    rxml.precision = singlePrec ? "F" : "D";
    writeLimeObject(0, 0, rxml, std::string("qudaEvecRecordXml"), SCIDAC_RECORD_XML);
    }

    // Open binary record with total declared size: (MB=0, ME=0) on the
    // master, a message of its own (MB=1, ME=1) on the other nodes.
    uint64_t total_bytes = (uint64_t)lsites * Nvec * vlen * elem_bytes;
    {
      int MB = master ? 0 : 1, ME = master ? 0 : 1;
      LimeRecordHeader *lh = limeCreateHeader(MB, ME,
                                              const_cast<char*>(SCIDAC_BINARY_DATA),
                                              total_bytes);
      GRID_ASSERT(limeWriteRecordHeader(lh, LimeW) >= 0);
      limeDestroyHeader(lh);
    }

    // Chunk buffer
    std::vector<float>  fbuf;
    std::vector<double> dbuf;
    if (singlePrec) fbuf.resize((size_t)chunk_sites * Nvec * vlen);
    else            dbuf.resize((size_t)chunk_sites * Nvec * vlen);

    uint32_t suma = 0, sumb = 0;

    for (int s0 = 0; s0 < lsites; s0 += chunk_sites) {
      int ns = std::min(chunk_sites, lsites - s0);

      // Fill chunk buffer (host byte order)
      for (int ds = 0; ds < ns; ds++) {
        Coordinate lcoor(nd);
        qudaIoSite(grid, cb, s0 + ds, lcoor);

        for (int i = 0; i < Nvec; i++) {
          sobj site_val;
          peekLocalSite(site_val, evecs[i], lcoor);
          // Cast sobj to flat SR array; layout matches QUDA's ordering.
          const SR *src  = reinterpret_cast<const SR *>(&site_val);
          int       base = (ds * Nvec + i) * vlen;
          if (singlePrec) {
            for (int k = 0; k < vlen; k++) fbuf[base + k] = (float)src[k];
          } else {
            for (int k = 0; k < vlen; k++) dbuf[base + k] = (double)src[k];
          }
        }
      }

      // Byte-swap to BE and checksum, one site at a time.
      // After this loop the chunk is in BE order, ready to write.
      for (int ds = 0; ds < ns; ds++) {
        Coordinate lcoor(nd);
        int64_t gs = qudaIoSite(grid, cb, s0 + ds, lcoor);

        int    base        = ds * Nvec * vlen;
        size_t site_bytes  = (size_t)Nvec * vlen * elem_bytes;

        if (singlePrec) {
          htobe32_v(&fbuf[base], (uint32_t)site_bytes);
          qudaAccumChecksum(gs, &fbuf[base], site_bytes, suma, sumb);
        } else {
          htobe64_v(&dbuf[base], (uint32_t)site_bytes);
          qudaAccumChecksum(gs, &dbuf[base], site_bytes, suma, sumb);
        }
      }

      // Write chunk (already big-endian)
      uint64_t chunk_bytes = (uint64_t)ns * Nvec * vlen * elem_bytes;
      int err;
      if (singlePrec)
        err = limeWriteRecordData(fbuf.data(), &chunk_bytes, LimeW);
      else
        err = limeWriteRecordData(dbuf.data(), &chunk_bytes, LimeW);
      GRID_ASSERT(err >= 0);
    }

    GRID_ASSERT(limeWriterCloseRecord(LimeW) >= 0);

    // Combine checksums over all nodes; master writes scidac-checksum (MB=0, ME=1)
    grid->GlobalXOR(suma);
    grid->GlobalXOR(sumb);
    if (master) writeChecksumRecord(suma, sumb, 0, 1);
  }

private:
  bool master = true;   // this rank writes QIO's master partfile (vol0000)

  // Write scidac-sitelist: lsites big-endian uint64_t global site indices.
  void writeSitelistRecord(GridBase *grid, int MB, int ME)
  {
    int nd     = grid->_ndimension;
    int lsites = grid->lSites();

    // The global index does not depend on the parity, so cb=0 is fine here.
    std::vector<uint64_t> sitelist(lsites);
    for (int s = 0; s < lsites; s++) {
      Coordinate lfull(nd);
      sitelist[s] = (uint64_t)qudaIoSite(grid, 0, s, lfull);
    }
    uint64_t nbytes = (uint64_t)lsites * sizeof(uint64_t);
    htobe64_v(sitelist.data(), (uint32_t)nbytes);
    writeLimeBinaryRecord(sitelist.data(), nbytes, MB, ME, SCIDAC_SITELIST);
  }

  // Write a raw binary LIME record in one shot.
  void writeLimeBinaryRecord(void *data, uint64_t nbytes,
                             int MB, int ME, const char *record_name)
  {
    int err;
    LimeRecordHeader *lh = limeCreateHeader(MB, ME,
                                            const_cast<char*>(record_name),
                                            nbytes);
    GRID_ASSERT(lh != NULL);
    err = limeWriteRecordHeader(lh, LimeW);  GRID_ASSERT(err >= 0);
    limeDestroyHeader(lh);
    err = limeWriteRecordData(data, &nbytes, LimeW);  GRID_ASSERT(err >= 0);
    err = limeWriterCloseRecord(LimeW);               GRID_ASSERT(err >= 0);
  }

  // Write a scidac-checksum record.
  void writeChecksumRecord(uint32_t suma, uint32_t sumb, int MB, int ME)
  {
    char tmp[16];
    scidacChecksum csum;
    csum.version = 1.0;
    snprintf(tmp, sizeof(tmp), "%08x", suma);  csum.suma = tmp;
    snprintf(tmp, sizeof(tmp), "%08x", sumb);  csum.sumb = tmp;
    writeLimeObject(MB, ME, csum, std::string("scidacChecksum"), SCIDAC_CHECKSUM);
  }
};

///////////////////////////////////////////////////////////////////////////////
// QudaPartfileReader
//
// Each MPI rank reads its own partfile.  Extends GridLimeReader.  As in QIO,
// only the master rank (vol0000) parses the XML records; it broadcasts the
// precision and parity to the others and verifies the combined checksum.
// Must be called by all ranks (collective).
//
// readQudaPartfile() accepts, in the master partfile:
//   (a) field record followed by an optional eigenvalue record (current);
//   (b) eigenvalue record followed by the field record (legacy, single rank).
// It reads partfiles written by QudaPartfileWriter and by QUDA itself
// (VectorIO::save with partfile=true, e.g. MILC save_partfile_scidac).
// If there is no eigenvalue record (QUDA never writes one), evals is
// filled with zeros.
///////////////////////////////////////////////////////////////////////////////
class QudaPartfileReader : public GridLimeReader {
public:
  // Open <stem>.vol<NNNN>, NNNN = QMP node number of this rank (qudaPartfileRank).
  void open(const std::string &stem, GridBase *grid)
  {
    int node = qudaPartfileRank(grid);
    master   = (node == 0);
    GridLimeReader::open(qudaPartfileName(stem, node));
  }

  // -------------------------------------------------------------------------
  // Top-level read.
  //
  // evecs must be pre-allocated to Nvec elements (Grid allocates the fields).
  // The file may hold Nfile >= Nvec vectors (e.g. a QUDA eigensolve with
  // n_conv > Nvec); the first Nvec are kept.
  // evals is resized to Nvec; filled from file or zeroed if not present
  // (files written by QUDA itself carry no eigenvalue record).
  // -------------------------------------------------------------------------
  template <class vobj>
  void readQudaPartfile(std::vector<Lattice<vobj>> &evecs,
                        std::vector<ComplexD>       &evals,
                        const std::string           &stem,
                        GridBase                    *grid,
                        int nColor, int nSpin)
  {
    int Nvec = (int)evecs.size();
    open(stem, grid);

    // hdr = {singlePrec, cb, have_evals, Nfile}, parsed by the master and
    // broadcast.
    int hdr[4] = {1, Even, 0, Nvec};
    if (master) readMasterHeader(evals, grid, nColor, nSpin, hdr);
    grid->Broadcast(qudaMasterRank(grid), (void *)hdr, sizeof(hdr));
    bool singlePrec = hdr[0];
    int  cb         = hdr[1];
    bool have_evals = hdr[2];
    int  Nfile      = hdr[3];

    if (Nfile < Nvec) {
      std::cerr << "QudaPartfileReader: file " << qudaPartfileName(stem, 0)
                << " holds " << Nfile << " vectors; " << Nvec
                << " requested. Aborting.\n";
      GRID_ASSERT(0);
    }

    // -- Read binary eigenvector data (chunked, de-interleave, checksum).
    readEvecBinaryData<vobj>(evecs, grid, cb, nColor, nSpin, singlePrec, Nfile);

    // -- Current order: optional eigenvalue record follows the field record.
    if (master && !have_evals) {
      if (seekLimeRecordOptional(SCIDAC_PRIVATE_RECORD_XML)) {
        readEvalBody(evals);
        have_evals = true;
      }
    }
    int he = have_evals;
    grid->Broadcast(qudaMasterRank(grid), (void *)&he, sizeof(he));
    if (he) {
      evals.resize(Nvec, ComplexD(0.0, 0.0));   // keep the first Nvec
      grid->Broadcast(qudaMasterRank(grid), (void *)evals.data(),
                      Nvec * sizeof(ComplexD));
    } else {
      if (master)
        std::cout << GridLogMessage
                  << "QudaPartfileReader: no global eigenvalue record in "
                  << qudaPartfileName(stem, 0)
                  << "; eigenvalues set to zero.\n";
      evals.assign(Nvec, ComplexD(0.0, 0.0));
    }

    close();
  }

private:
  bool master = true;   // this rank reads QIO's master partfile (vol0000)

  // -------------------------------------------------------------------------
  // Master only: read the file header and the field record's XML, leaving
  // the reader just before the field's binary data.  Fills
  // hdr = {singlePrec, cb, have_evals, Nfile}; reads evals if they come first.
  //
  // Accepts files written by Grid (QudaPartfileWriter) and by QUDA
  // (write_spinor_field via QIO), whose record XML differ.
  // -------------------------------------------------------------------------
  void readMasterHeader(std::vector<ComplexD> &evals, GridBase *grid,
                        int nColor, int nSpin, int hdr[4])
  {
    // -- Message 1: check the lattice dimensions in the private file XML.
    // Sitelist and file-xml follow; they are skipped by the next scan.
    {
      std::string pfx;
      readLimeObject(pfx, std::string(SCIDAC_PRIVATE_FILE_XML));
      std::istringstream ds(qudaXmlTag(pfx, "dims"));
      Coordinate gdims = grid->GlobalDimensions();
      bool ok = true;
      for (int d = 0; d < grid->_ndimension; d++) {
        int n = -1;
        ds >> n;
        if (n != gdims[d]) ok = false;
      }
      if (!ok) {
        std::cerr << "QudaPartfileReader: lattice dimensions in file ("
                  << qudaXmlTag(pfx, "dims") << ") do not match grid "
                  << gdims << ". Aborting.\n";
        GRID_ASSERT(0);
      }
    }

    // -- Message 2: private-record-xml determines the record type.
    std::string priv2_xml;
    readLimeObject(priv2_xml, std::string(SCIDAC_PRIVATE_RECORD_XML));
    scidacRecord prec2;
    {
      XmlReader RD(priv2_xml, true, "");
      read(RD, "scidacRecord", prec2);
    }

    bool have_evals = false;
    if (prec2.recordtype == GRID_IO_GLOBAL) {
      // -- Legacy order: eigenvalue record precedes the field record.
      readEvalBody(evals);
      have_evals = true;

      std::string priv3_xml;
      readLimeObject(priv3_xml, std::string(SCIDAC_PRIVATE_RECORD_XML));
      XmlReader RD(priv3_xml, true, "");
      read(RD, "scidacRecord", prec2);
    }

    GRID_ASSERT(prec2.recordtype == GRID_IO_FIELD);
    bool singlePrec = (prec2.precision == "F" || prec2.precision == "f");
    int  elem_bytes = singlePrec ? (int)sizeof(float) : (int)sizeof(double);
    int  vlen       = nSpin * nColor * 2;
    if (prec2.colors != nColor || prec2.spins != nSpin ||
        prec2.typesize != vlen * elem_bytes || prec2.datacount < 1) {
      std::cerr << "QudaPartfileReader: field record has colors=" << prec2.colors
                << " spins=" << prec2.spins << " typesize=" << prec2.typesize
                << " datacount=" << prec2.datacount << "; expected colors="
                << nColor << " spins=" << nSpin << " typesize="
                << vlen * elem_bytes << ". Aborting.\n";
      GRID_ASSERT(0);
    }

    // -- Field record-xml: subset and parity.  Grid writes them at the top
    // level of <qudaEvecRecordXml>; QUDA writes them inside <info>.
    int cb = Even;
    {
      std::string rxml_str;
      readLimeObject(rxml_str, std::string(SCIDAC_RECORD_XML));
      std::string subset = qudaXmlTag(rxml_str, "subset");
      std::string parity = qudaXmlTag(rxml_str, "parity");
      if (grid->_isCheckerBoarded) {
        if (subset != "parity" || (parity != "even" && parity != "odd")) {
          std::cerr << "QudaPartfileReader: red-black grid requires subset=parity"
                    << " and parity=even|odd; file has subset=" << subset
                    << " parity=" << parity << ". Aborting.\n";
          GRID_ASSERT(0);
        }
        cb = (parity == "even") ? Even : Odd;
      } else if (subset != "full") {
        std::cerr << "QudaPartfileReader: full grid requires subset=full;"
                  << " file has subset=" << subset << ". Aborting.\n";
        GRID_ASSERT(0);
      }
    }

    hdr[0] = singlePrec;
    hdr[1] = cb;
    hdr[2] = have_evals;
    hdr[3] = prec2.datacount;
  }

  // -------------------------------------------------------------------------
  // Read the body of a global eigenvalue record (all values in the record).
  // Called after private-record-xml has already been consumed.
  // Reads: record-xml, binary data, checksum.
  // -------------------------------------------------------------------------
  void readEvalBody(std::vector<ComplexD> &evals)
  {
    // record-xml (advance past it; content is informational)
    {
      std::string tmp;
      readLimeObject(tmp, std::string(SCIDAC_RECORD_XML));
    }

    // binary data: [re0, im0, re1, im1, ...] in big-endian doubles
    seekLimeRecord(SCIDAC_BINARY_DATA);
    uint64_t nbytes = limeReaderBytes(LimeR);
    int ne = (int)(nbytes / (2 * sizeof(double)));
    std::vector<double> buf(2 * ne);
    limeReaderReadData(buf.data(), &nbytes, LimeR);

    // Checksum on BE data (before byte-swap)
    uint32_t suma = 0, sumb = 0;
    qudaAccumChecksum(0, buf.data(), nbytes, suma, sumb);

    // Byte-swap to host
    be64toh_v(buf.data(), nbytes);

    evals.resize(ne);
    for (int i = 0; i < ne; i++) {
      evals[i] = ComplexD(buf[2*i], buf[2*i+1]);
    }

    // Verify checksum
    verifyChecksum(suma, sumb, "eigenvalue");
  }

  // -------------------------------------------------------------------------
  // Read and de-interleave eigenvector binary data.
  // Called after field record's record-xml has been consumed.
  // Each site holds Nfile vectors; the first evecs.size() are kept.
  // Reads: binary data (chunked), then verifies checksum.
  // -------------------------------------------------------------------------
  template <class vobj>
  void readEvecBinaryData(std::vector<Lattice<vobj>> &evecs,
                          GridBase *grid, int cb,
                          int nColor, int nSpin,
                          bool singlePrec, int Nfile,
                          int chunk_sites = 1024)
  {
    typedef typename vobj::scalar_object sobj;
    typedef typename sobj::scalar_type   SC;
    typedef typename SC::value_type     SR;

    int Nvec   = (int)evecs.size();
    int nd     = grid->_ndimension;
    int vlen   = nSpin * nColor * 2;
    int lsites = grid->lSites();                // half volume on red-black grid

    for (int i = 0; i < Nvec; i++) evecs[i].Checkerboard() = cb;

    assert((int)(sizeof(sobj) / sizeof(SR)) == vlen);

    int elem_bytes = singlePrec ? (int)sizeof(float) : (int)sizeof(double);

    // Advance to binary record
    seekLimeRecord(SCIDAC_BINARY_DATA);

    std::vector<float>  fbuf;
    std::vector<double> dbuf;
    if (singlePrec) fbuf.resize((size_t)chunk_sites * Nfile * vlen);
    else            dbuf.resize((size_t)chunk_sites * Nfile * vlen);

    uint32_t suma = 0, sumb = 0;

    for (int s0 = 0; s0 < lsites; s0 += chunk_sites) {
      int ns = std::min(chunk_sites, lsites - s0);
      uint64_t chunk_bytes = (uint64_t)ns * Nfile * vlen * elem_bytes;

      // Read chunk (big-endian as stored in file)
      if (singlePrec)
        limeReaderReadData(fbuf.data(), &chunk_bytes, LimeR);
      else
        limeReaderReadData(dbuf.data(), &chunk_bytes, LimeR);

      // Process site by site: checksum on BE, byte-swap, poke into Grid.
      for (int ds = 0; ds < ns; ds++) {
        Coordinate lcoor(nd);
        int64_t gs = qudaIoSite(grid, cb, s0 + ds, lcoor);

        size_t base        = (size_t)ds * Nfile * vlen;
        size_t site_bytes  = (size_t)Nfile * vlen * elem_bytes;

        // Checksum on big-endian data (all Nfile vectors at the site)
        if (singlePrec)
          qudaAccumChecksum(gs, &fbuf[base], site_bytes, suma, sumb);
        else
          qudaAccumChecksum(gs, &dbuf[base], site_bytes, suma, sumb);

        // Byte-swap to host order
        if (singlePrec)
          be32toh_v(&fbuf[base], (uint64_t)site_bytes);
        else
          be64toh_v(&dbuf[base], (uint64_t)site_bytes);

        // De-interleave: poke each vector's data into the Grid lattice.
        // sobj layout (spin-major, color-minor, re-im) matches file layout.
        for (int i = 0; i < Nvec; i++) {
          sobj site_val;
          SR *dst         = reinterpret_cast<SR *>(&site_val);
          size_t vec_base = base + (size_t)i * vlen;
          if (singlePrec) {
            for (int k = 0; k < vlen; k++) dst[k] = (SR)fbuf[vec_base + k];
          } else {
            for (int k = 0; k < vlen; k++) dst[k] = (SR)dbuf[vec_base + k];
          }
          pokeLocalSite(site_val, evecs[i], lcoor);
        }
      }
    }

    // Combine checksums over all nodes; master reads and verifies scidac-checksum
    grid->GlobalXOR(suma);
    grid->GlobalXOR(sumb);
    if (master) verifyChecksum(suma, sumb, "eigenvector");
  }

  // Advance LimeR to the first record whose type matches record_name.
  // Starts from the record AFTER the current position (calls NextRecord first).
  void seekLimeRecord(const char *record_name)
  {
    while (limeReaderNextRecord(LimeR) == LIME_SUCCESS) {
      if (std::string(limeReaderType(LimeR)) == std::string(record_name))
        return;
    }
    std::cerr << "QudaPartfileReader: record not found: " << record_name << "\n";
    GRID_ASSERT(0);
  }

  // As seekLimeRecord, but returns false at end of file instead of aborting.
  bool seekLimeRecordOptional(const char *record_name)
  {
    while (limeReaderNextRecord(LimeR) == LIME_SUCCESS) {
      if (std::string(limeReaderType(LimeR)) == std::string(record_name))
        return true;
    }
    return false;
  }

  // Seek to a binary record and read all nbytes from it.
  void seekAndReadBinary(const char *record_name, void *buf, uint64_t nbytes)
  {
    seekLimeRecord(record_name);
    limeReaderReadData(buf, &nbytes, LimeR);
  }

  // Read the next scidac-checksum record and verify against computed values.
  void verifyChecksum(uint32_t comp_a, uint32_t comp_b,
                      const std::string &context)
  {
    std::string cxml_str;
    readLimeObject(cxml_str, std::string(SCIDAC_CHECKSUM));
    scidacChecksum csum;
    XmlReader CR(cxml_str, true, "");
    read(CR, "scidacChecksum", csum);

    uint32_t ref_a = (uint32_t)stoull(csum.suma, 0, 16);
    uint32_t ref_b = (uint32_t)stoull(csum.sumb, 0, 16);

    if (comp_a != ref_a || comp_b != ref_b) {
      std::cerr << "QudaPartfileReader: " << context << " checksum mismatch: "
                << "computed (" << std::hex << comp_a << "," << comp_b << ") "
                << "file ("     << ref_a    << "," << ref_b    << ")"
                << std::dec << "\n";
      GRID_ASSERT(0);
    }
    std::cout << GridLogMessage
              << "QudaPartfileReader: " << context << " checksum OK ("
              << std::hex << comp_a << "," << comp_b << ")" << std::dec << "\n";
  }
};

///////////////////////////////////////////////////////////////////////////////
// Convenience free functions
///////////////////////////////////////////////////////////////////////////////

// Write a QUDA eigenpack partfile set.
// evecs: Nvec Grid fields, either full-lattice or single-parity (red-black).
// evals: eigenvalues (e.g. ComplexD as in AdaptorEigenPackMILC::eval).
// nColor=3, nSpin=1 for staggered; nSpin=4 for Wilson.
template <class vobj>
void writeQudaEigenpackPartfile(const std::string           &stem,
                                std::vector<Lattice<vobj>>  &evecs,
                                const std::vector<ComplexD> &evals,
                                GridBase                    *grid,
                                int nColor = 3, int nSpin = 1,
                                bool singlePrec = true)
{
  QudaPartfileWriter wr;
  wr.open(stem, grid);
  wr.writeFileHeader(grid, (int)evecs.size(), singlePrec ? "F" : "D");
  wr.writeEvecFieldRecord(evecs, grid, nColor, nSpin, singlePrec);
  wr.writeGlobalEvalRecord(evals);
  wr.close();
}

// Read a QUDA eigenpack partfile set.
// evecs must be pre-allocated to Nvec elements; evals is resized to Nvec.
template <class vobj>
void readQudaEigenpackPartfile(const std::string           &stem,
                               std::vector<Lattice<vobj>>  &evecs,
                               std::vector<ComplexD>       &evals,
                               GridBase                    *grid,
                               int nColor = 3, int nSpin = 1)
{
  QudaPartfileReader rd;
  rd.readQudaPartfile(evecs, evals, stem, grid, nColor, nSpin);
}

NAMESPACE_END(Grid);

#endif  // HAVE_LIME
#endif  // GRID_QUDA_PARTFILE_IO_H
