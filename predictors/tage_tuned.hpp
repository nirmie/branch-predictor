// Tuned TAGE: hyperparameters selected by genetic-algorithm design-space
// exploration on gcc_test_trace.gz. Template args are:
//
//   <LOGLB=6, NUMG=7, LOGG=11, LOGB=11, TAGW=12, GHIST=100, LOGP1=13, GHIST1=10>
//
// versus the reference tage<> default (6, 8, 11, 12, 11, 100, 14, 6).
// On gcc_test_trace.gz (1M warmup / 40M measure):
//   tage<>       : MPKI=6.198  E=1542 fJ/instr  VFS=0.7653
//   tage_tuned<> : MPKI=6.095  E=1156 fJ/instr  VFS=0.7825  (+2.25% VFS)

// Requires tage.hpp to be included first (branch_predictor.hpp does this).

template<int = 0>
struct tage_tuned : tage<6, 7, 11, 11, 12, 100, 13, 10> {};
