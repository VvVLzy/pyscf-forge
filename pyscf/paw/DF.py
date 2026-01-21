import numpy



from pyscf.pbc.lib.kpts_helper import unique
from pyscf.pbc.df import GDF, incore
from pyscf.lib import logger

import os

class PAWDF(GDF):
    def build(self, j_only=None, with_j3c=True, kpts_band=None):
        if j_only is not None:
            self._j_only = j_only
        if self.kpts_band is not None:
            self.kpts_band = numpy.reshape(self.kpts_band, (-1,3))
        if kpts_band is not None:
            kpts_band = numpy.reshape(kpts_band, (-1,3))
            if self.kpts_band is None:
                self.kpts_band = kpts_band
            else:
                self.kpts_band = unique(numpy.vstack((self.kpts_band,kpts_band)))[0]

        self.check_sanity()
        self.dump_flags()

        self.auxcell = incore.make_auxcell(self.cell, self.auxbasis)

        if with_j3c and self._cderi_to_save is not None:
            if isinstance(self._cderi_to_save, str):
                cderi = self._cderi_to_save
            else:
                cderi = self._cderi_to_save.name
            if isinstance(self._cderi, str):
                if self._cderi == cderi and os.path.isfile(cderi):
                    logger.warn(self, 'File %s (specified by ._cderi) is '
                                'overwritten by GDF initialization.', cderi)
                    os.remove(cderi)
                else:
                    logger.warn(self, 'Value of ._cderi is ignored. '
                                'DF integrals will be saved in file %s .', cderi)
            self._cderi = cderi
            t1 = (logger.process_clock(), logger.perf_counter())
            self._make_j3c(self.cell, self.auxcell, None, cderi)
            t1 = logger.timer_debug1(self, 'j3c', *t1)
        return self