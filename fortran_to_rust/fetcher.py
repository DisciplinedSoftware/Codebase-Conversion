"""Fetch and cache BLAS (and other Fortran library) source code."""

from __future__ import annotations

import os
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console

console = Console()

# ---------------------------------------------------------------------------
# Known BLAS function names (reference BLAS Level 1-3)
# ---------------------------------------------------------------------------
BLAS_FUNCTIONS: List[str] = [
    # Level 1
    "dasum", "daxpy", "dcopy", "ddot", "dnrm2", "drot", "drotg", "drotm",
    "drotmg", "dscal", "dsdot", "dswap", "dzasum", "dznrm2", "idamax",
    # Level 2
    "dgbmv", "dgemv", "dger", "dsbmv", "dspmv", "dspr", "dspr2", "dsymv",
    "dsyr", "dsyr2", "dtbmv", "dtbsv", "dtpmv", "dtpsv", "dtrmv", "dtrsv",
    # Level 3
    "dgemm", "dsymm", "dsyr2k", "dsyrk", "dtrmm", "dtrsm",
]

# A small illustrative subset used in "sample of functions" mode
BLAS_SAMPLE: List[str] = [
    "dgemm", "dgemv", "ddot", "dscal", "daxpy",
    "dnrm2", "dcopy", "dger", "dsymm", "dtrsm",
]

# ---------------------------------------------------------------------------
# Download sources
# ---------------------------------------------------------------------------
# Primary: individual files from the netlib mirror
_NETLIB_BASE = "http://www.netlib.org/blas/"
_BLAS_TARBALL = "http://www.netlib.org/blas/blas-3.11.0.tgz"
_LAPACK_BLAS_GITHUB_RAW = (
    "https://raw.githubusercontent.com/Reference-LAPACK/lapack/master/BLAS/SRC/"
)


def _cache_dir(output_dir: Path) -> Path:
    d = output_dir / "fortran" / "blas"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _try_download_file(url: str, dest: Path) -> bool:
    """Return True if file was downloaded successfully."""
    try:
        urllib.request.urlretrieve(url, dest)  # noqa: S310
        return dest.exists() and dest.stat().st_size > 100
    except Exception:
        return False


def _try_download_tarball(output_dir: Path) -> bool:
    """Try to download the full BLAS tarball and extract it."""
    tmp_tar = output_dir / "blas.tgz"
    console.print(f"  [dim]Trying tarball: {_BLAS_TARBALL}[/dim]")
    if not _try_download_file(_BLAS_TARBALL, tmp_tar):
        return False
    try:
        with tarfile.open(tmp_tar) as t:
            t.extractall(output_dir)  # noqa: S202
        # Move extracted .f files into cache dir
        cache = _cache_dir(output_dir)
        for extracted in output_dir.rglob("*.f"):
            shutil.copy2(extracted, cache / extracted.name)
        tmp_tar.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def fetch_blas(
    output_dir: Path,
    functions: Optional[List[str]] = None,
    *,
    progress_callback=None,
) -> Dict[str, Path]:
    """Download BLAS Fortran source files.

    Returns a mapping of function_name -> path_to_.f_file.
    Falls back to an embedded copy of dgemm if the network is unavailable.
    """
    cache = _cache_dir(output_dir)
    wanted = functions or BLAS_FUNCTIONS
    result: Dict[str, Path] = {}

    # Check which files we already have cached
    missing = [fn for fn in wanted if not (cache / f"{fn}.f").exists()]

    if missing:
        console.print(f"  [cyan]Fetching {len(missing)} BLAS source file(s)…[/cyan]")

        # Try individual files from GitHub first (more reliable)
        for fn in list(missing):
            dest = cache / f"{fn}.f"
            url = _LAPACK_BLAS_GITHUB_RAW + f"{fn.upper()}.f"
            if progress_callback:
                progress_callback(f"Downloading {fn}.f")
            if _try_download_file(url, dest):
                missing.remove(fn)
                console.print(f"    [green]✓[/green] {fn}.f")
            else:
                # Try netlib directly
                url2 = _NETLIB_BASE + f"{fn}.f"
                if _try_download_file(url2, dest):
                    missing.remove(fn)
                    console.print(f"    [green]✓[/green] {fn}.f (netlib)")

        # If still missing, try the tarball
        if missing:
            console.print("  [yellow]Some files missing, trying tarball…[/yellow]")
            _try_download_tarball(output_dir)
            missing = [fn for fn in wanted if not (cache / f"{fn}.f").exists()]

        # For any still missing, write embedded fallback
        for fn in missing:
            src = _get_embedded(fn)
            if src:
                dest = cache / f"{fn}.f"
                dest.write_text(src)
                console.print(f"    [yellow]⚠[/yellow]  {fn}.f (embedded fallback)")
            else:
                console.print(f"    [red]✗[/red] {fn}.f not available")

    for fn in wanted:
        p = cache / f"{fn}.f"
        if p.exists():
            result[fn] = p

    return result


# ---------------------------------------------------------------------------
# Embedded Fortran source (used only when network is unavailable)
# ---------------------------------------------------------------------------
_EMBEDDED: Dict[str, str] = {}


def _get_embedded(name: str) -> Optional[str]:
    return _EMBEDDED.get(name.lower())


# We embed only dgemm as the primary demo target.
_EMBEDDED["dgemm"] = r"""
*> \brief \b DGEMM
*
*  -- Reference BLAS level3 routine --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd.     --
*
      SUBROUTINE DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
*
*  Purpose
*  =======
*
*  DGEMM performs one of the matrix-matrix operations
*
*     C := alpha*op( A )*op( B ) + beta*C,
*
*  where  op( X ) is one of
*
*     op( X ) = X   or   op( X ) = X**T,
*
*  alpha and beta are scalars, and A, B and C are matrices, with op( A )
*  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
*
*  Arguments
*  =========
*
*  TRANSA - CHARACTER*1.
*           On entry, TRANSA specifies the form of op( A ) to be used in
*           the matrix multiplication as follows:
*
*              TRANSA = 'N' or 'n',  op( A ) = A.
*              TRANSA = 'T' or 't',  op( A ) = A**T.
*              TRANSA = 'C' or 'c',  op( A ) = A**T.
*
*  TRANSB - CHARACTER*1.
*           On entry, TRANSB specifies the form of op( B ) to be used in
*           the matrix multiplication as follows:
*
*              TRANSB = 'N' or 'n',  op( B ) = B.
*              TRANSB = 'T' or 't',  op( B ) = B**T.
*              TRANSB = 'C' or 'c',  op( B ) = B**T.
*
*  M      - INTEGER.
*           On entry,  M  specifies  the number  of rows  of the  matrix
*           op( A )  and of the  matrix  C.  M  must  be at least  zero.
*
*  N      - INTEGER.
*           On entry,  N  specifies the number  of columns of the matrix
*           op( B ) and the number of columns of the matrix C. N must be
*           at least zero.
*
*  K      - INTEGER.
*           On entry,  K  specifies  the number of columns of the matrix
*           op( A ) and the number of rows of the matrix op( B ). K must
*           be at least zero.
*
*  ALPHA  - DOUBLE PRECISION.
*           On entry, ALPHA specifies the scalar alpha.
*
*  A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
*           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
*           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
*           part of the array  A  must contain the matrix  A.
*           Before entry with TRANSA = 'T' or 't' or 'C' or 'c',
*           the leading  k by m  part of the array  A  must contain  the
*           matrix A.
*
*  LDA    - INTEGER.
*           On entry, LDA specifies the first dimension of A as declared
*           in the calling (sub) program. When  TRANSA = 'N' or 'n' then
*           LDA must be at least  max( 1, m ), otherwise  LDA must be at
*           least  max( 1, k ).
*
*  B      - DOUBLE PRECISION array of DIMENSION ( LDB, kb ), where kb is
*           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
*           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
*           part of the array  B  must contain the matrix  B.
*           Before entry with TRANSB = 'T' or 't' or 'C' or 'c',
*           the leading  n by k  part of the array  B  must contain  the
*           matrix B.
*
*  LDB    - INTEGER.
*           On entry, LDB specifies the first dimension of B as declared
*           in the calling (sub) program. When  TRANSB = 'N' or 'n' then
*           LDB must be at least  max( 1, k ), otherwise  LDB must be at
*           least  max( 1, n ).
*
*  BETA   - DOUBLE PRECISION.
*           On entry,  BETA  specifies the scalar  beta.  When  BETA  is
*           supplied as zero then C need not be set on input.
*
*  C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
*           Before entry, the leading  m by n  part of the array  C must
*           contain the matrix  C,  except when  beta  is zero, in which
*           case C need not be set on entry.
*           On exit, the array  C  is overwritten by the  m by n  matrix
*           ( alpha*op( A )*op( B ) + beta*C ).
*
*  LDC    - INTEGER.
*           On entry, LDC specifies the first dimension of C as declared
*           in the calling (sub) program.  LDC  must  be  at  least
*           max( 1, m ).
*
      DOUBLE PRECISION ALPHA,BETA
      INTEGER K,LDA,LDB,LDC,M,N
      CHARACTER TRANSA,TRANSB
*
      DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
*     ..
*     .. External Functions ..
      LOGICAL LSAME
      EXTERNAL LSAME
*     ..
*     .. Local Scalars ..
      DOUBLE PRECISION TEMP
      INTEGER I,INFO,J,L,NROWA,NROWB
      LOGICAL NOTA,NOTB
*     ..
*     .. Parameters ..
      DOUBLE PRECISION ONE,ZERO
      PARAMETER (ONE=1.0D+0,ZERO=0.0D+0)
*
*     Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not
*     transposed and set  NROWA and NROWB  as the number of rows and
*     columns of  A  and the number of rows of  B  respectively.
*
      NOTA = LSAME(TRANSA,'N')
      NOTB = LSAME(TRANSB,'N')
      IF (NOTA) THEN
          NROWA = M
      ELSE
          NROWA = K
      END IF
      IF (NOTB) THEN
          NROWB = K
      ELSE
          NROWB = N
      END IF
*
*     Test the input parameters.
*
      INFO = 0
      IF ((.NOT.NOTA) .AND. (.NOT.LSAME(TRANSA,'C')) .AND.
     +    (.NOT.LSAME(TRANSA,'T'))) THEN
          INFO = 1
      ELSE IF ((.NOT.NOTB) .AND. (.NOT.LSAME(TRANSB,'C')) .AND.
     +         (.NOT.LSAME(TRANSB,'T'))) THEN
          INFO = 2
      ELSE IF (M.LT.0) THEN
          INFO = 3
      ELSE IF (N.LT.0) THEN
          INFO = 4
      ELSE IF (K.LT.0) THEN
          INFO = 5
      ELSE IF (LDA.LT.MAX(1,NROWA)) THEN
          INFO = 8
      ELSE IF (LDB.LT.MAX(1,NROWB)) THEN
          INFO = 10
      ELSE IF (LDC.LT.MAX(1,M)) THEN
          INFO = 13
      END IF
      IF (INFO.NE.0) THEN
          CALL XERBLA('DGEMM ',INFO)
          RETURN
      END IF
*
*     Quick return if possible.
*
      IF ((M.EQ.0) .OR. (N.EQ.0) .OR.
     +    (((ALPHA.EQ.ZERO).OR.(K.EQ.0)).AND. (BETA.EQ.ONE))) RETURN
*
*     And if  alpha.eq.zero.
*
      IF (ALPHA.EQ.ZERO) THEN
          IF (BETA.EQ.ZERO) THEN
              DO 20 J = 1,N
                  DO 10 I = 1,M
                      C(I,J) = ZERO
   10             CONTINUE
   20         CONTINUE
          ELSE
              DO 40 J = 1,N
                  DO 30 I = 1,M
                      C(I,J) = BETA*C(I,J)
   30             CONTINUE
   40         CONTINUE
          END IF
          RETURN
      END IF
*
*     Start the operations.
*
      IF (NOTB) THEN
          IF (NOTA) THEN
*
*           Form  C := alpha*A*B + beta*C.
*
              DO 90 J = 1,N
                  IF (BETA.EQ.ZERO) THEN
                      DO 50 I = 1,M
                          C(I,J) = ZERO
   50                 CONTINUE
                  ELSE IF (BETA.NE.ONE) THEN
                      DO 60 I = 1,M
                          C(I,J) = BETA*C(I,J)
   60                 CONTINUE
                  END IF
                  DO 80 L = 1,K
                      TEMP = ALPHA*B(L,J)
                      DO 70 I = 1,M
                          C(I,J) = C(I,J) + TEMP*A(I,L)
   70                 CONTINUE
   80             CONTINUE
   90         CONTINUE
          ELSE
*
*           Form  C := alpha*A**T*B + beta*C
*
              DO 120 J = 1,N
                  DO 110 I = 1,M
                      TEMP = ZERO
                      DO 100 L = 1,K
                          TEMP = TEMP + A(L,I)*B(L,J)
  100                 CONTINUE
                      IF (BETA.EQ.ZERO) THEN
                          C(I,J) = ALPHA*TEMP
                      ELSE
                          C(I,J) = ALPHA*TEMP + BETA*C(I,J)
                      END IF
  110             CONTINUE
  120         CONTINUE
          END IF
      ELSE
          IF (NOTA) THEN
*
*           Form  C := alpha*A*B**T + beta*C
*
              DO 150 J = 1,N
                  IF (BETA.EQ.ZERO) THEN
                      DO 130 I = 1,M
                          C(I,J) = ZERO
  130                 CONTINUE
                  ELSE IF (BETA.NE.ONE) THEN
                      DO 140 I = 1,M
                          C(I,J) = BETA*C(I,J)
  140                 CONTINUE
                  END IF
                  DO 160 L = 1,K
                      TEMP = ALPHA*B(J,L)
                      DO 155 I = 1,M
                          C(I,J) = C(I,J) + TEMP*A(I,L)
  155                 CONTINUE
  160             CONTINUE
  150         CONTINUE
          ELSE
*
*           Form  C := alpha*A**T*B**T + beta*C
*
              DO 190 J = 1,N
                  DO 180 I = 1,M
                      TEMP = ZERO
                      DO 170 L = 1,K
                          TEMP = TEMP + A(L,I)*B(J,L)
  170                 CONTINUE
                      IF (BETA.EQ.ZERO) THEN
                          C(I,J) = ALPHA*TEMP
                      ELSE
                          C(I,J) = ALPHA*TEMP + BETA*C(I,J)
                      END IF
  180             CONTINUE
  190         CONTINUE
          END IF
      END IF
*
      RETURN
*
*     End of DGEMM.
*
      END
"""

_EMBEDDED["lsame"] = r"""
      LOGICAL FUNCTION LSAME(CA,CB)
*     .. Scalar Arguments ..
      CHARACTER CA,CB
*     ..
*  Purpose
*  =======
*  LSAME returns .TRUE. if CA is the same letter as CB regardless of case.
*     .. Intrinsic Functions ..
      INTRINSIC ICHAR
*     .. Local Scalars ..
      INTEGER INTA,INTB,ZCODE
*
*     Test if the characters are equal
*
      LSAME = CA .EQ. CB
      IF (LSAME) RETURN
*
*     Now test for equivalence if both characters are alphabetic.
*
      ZCODE = ICHAR('Z')
*
*     Use 'Z' rather than 'A' so that ASCII and EBCDIC both work.
*
      INTA = ICHAR(CA)
      INTB = ICHAR(CB)
*
      IF (ZCODE.EQ.90 .OR. ZCODE.EQ.122) THEN
*
*        ASCII is assumed - ZCODE is the ASCII code of either lower or
*        upper case 'Z'.
*
          IF (INTA.GE.97 .AND. INTA.LE.122) INTA = INTA - 32
          IF (INTB.GE.97 .AND. INTB.LE.122) INTB = INTB - 32
*
      ELSE IF (ZCODE.EQ.233 .OR. ZCODE.EQ.169) THEN
*
*        EBCDIC is assumed - ZCODE is the EBCDIC code of either lower or
*        upper case 'Z'.
*
          IF (INTA.GE.129 .AND. INTA.LE.137 .OR.
     +        INTA.GE.145 .AND. INTA.LE.153 .OR.
     +        INTA.GE.162 .AND. INTA.LE.169) INTA = INTA + 64
          IF (INTB.GE.129 .AND. INTB.LE.137 .OR.
     +        INTB.GE.145 .AND. INTB.LE.153 .OR.
     +        INTB.GE.162 .AND. INTB.LE.169) INTB = INTB + 64
*
      ELSE IF (ZCODE.EQ.218 .OR. ZCODE.EQ.250) THEN
*
*        ASCII is assumed, on Prime machines - ZCODE is the ASCII code
*        plus 128 of either upper or lower case 'Z'.
*
          IF (INTA.GE.225 .AND. INTA.LE.250) INTA = INTA - 32
          IF (INTB.GE.225 .AND. INTB.LE.250) INTB = INTB - 32
      END IF
      LSAME = INTA .EQ. INTB
      RETURN
      END
"""

_EMBEDDED["xerbla"] = r"""
      SUBROUTINE XERBLA( SRNAME, INFO )
*     .. Scalar Arguments ..
      CHARACTER*(*)          SRNAME
      INTEGER                INFO
*
*  Purpose
*  =======
*
*  XERBLA  is an error handler for the BLAS routines.
*  It is called by the BLAS routines if an input parameter has an
*  invalid value.
*
*     .. Executable Statements ..
*
      WRITE( *, FMT = 9999 )SRNAME, INFO
*
      STOP
*
 9999 FORMAT( ' ** On entry to ', A, ' parameter number ', I2,
     $      ' had an illegal value' )
      END
"""
