#include "../cbp.hpp"
#include "../harcom.hpp"

using namespace hcm;


// Tournament predictor combining a bimodal (PC-indexed) component and a
// gshare (PC + global history) component, with a PC-indexed 2-bit meta
// selector choosing between them per branch.
//   meta top bit = 1 -> use gshare prediction
//   meta top bit = 0 -> use bimodal prediction
// P1 returns the (fast) bimodal prediction; P2 returns the tournament
// prediction. The two may disagree when meta selects gshare, in which case
// P2 overrides P1 (a cycle bubble, same mechanism as gshare.hpp / bimodal.hpp).
//
// Template parameters:
//   LOGLB  : log2(cache line bytes). LINEINST = 2^(LOGLB-2) predictions/cycle.
//   LOGG   : log2(gshare entries). Index bits per bank = LOGG - log2(LINEINST).
//   GHIST  : global history length (bits).
//   LOGB   : log2(bimodal entries).
//   LOGM   : log2(meta selector entries).
template<u64 LOGLB=6, u64 LOGG=15, u64 GHIST=12, u64 LOGB=14, u64 LOGM=14>
struct tournament : predictor {
    static_assert(LOGLB>2);
    static constexpr u64 LOGLINEINST = LOGLB-2;
    static constexpr u64 LINEINST = 1<<LOGLINEINST;
    static_assert(LOGG > LOGLINEINST);
    static_assert(LOGB > LOGLINEINST);
    static_assert(LOGM > LOGLINEINST);
    static constexpr u64 idx_g_bits = LOGG - LOGLINEINST;
    static constexpr u64 idx_b_bits = LOGB - LOGLINEINST;
    static constexpr u64 idx_m_bits = LOGM - LOGLINEINST;

    reg<GHIST> global_history;
    reg<1> true_block = 1;

    reg<idx_g_bits> index_g;
    reg<idx_b_bits> index_b;
    reg<idx_m_bits> index_m;

    arr<reg<1>,LINEINST> g_hi; // gshare bits read from table
    arr<reg<1>,LINEINST> b_hi; // bimodal bits read from table
    arr<reg<1>,LINEINST> m_hi; // meta bits read from table

    reg<LINEINST> p_g;     // gshare predictions
    reg<LINEINST> p_b;     // bimodal predictions (also = P1 output)
    reg<LINEINST> p_m;     // meta selector bits
    reg<LINEINST> p_final; // tournament predictions (= P2 output)

    // simulation artifacts, hardware cost may not be real
    u64 num_branch = 0;
    u64 block_size = 0;
    arr<reg<LOGLINEINST>,LINEINST> branch_offset;
    arr<reg<1>,LINEINST> branch_dir;
    reg<LINEINST> block_entry; // one-hot vector

    ram<val<1>,(1<<idx_g_bits)> table_g_hi[LINEINST];
    ram<val<1>,(1<<idx_b_bits)> table_b_hi[LINEINST];
    ram<val<1>,(1<<idx_m_bits)> table_m_hi[LINEINST];

    zone UPDATE_ONLY;
    // hysteresis bits (0 = weak, 1 = strong)
    ram<val<1>,(1<<idx_g_bits)> table_g_lo[LINEINST];
    ram<val<1>,(1<<idx_b_bits)> table_b_lo[LINEINST];
    ram<val<1>,(1<<idx_m_bits)> table_m_lo[LINEINST];

    void new_block(val<64> inst_pc)
    {
        val<LOGLINEINST> offset = inst_pc >> 2;
        block_entry = offset.decode().concat();
        block_entry.fanout(hard<4*LINEINST>{});
        block_size = 1;
    }

    val<1> predict1(val<64> inst_pc)
    {
        inst_pc.fanout(hard<2>{});
        new_block(inst_pc);
        val<std::max({idx_g_bits, idx_b_bits, idx_m_bits, GHIST})> lineaddr = inst_pc >> LOGLB;
        lineaddr.fanout(hard<3>{});

        if constexpr (GHIST <= idx_g_bits) {
            index_g = val<idx_g_bits>{lineaddr} ^ (val<idx_g_bits>{global_history} << (idx_g_bits - GHIST));
        } else {
            index_g = global_history.make_array(val<idx_g_bits>{}).append(val<idx_g_bits>{lineaddr}).fold_xor();
        }
        index_b = val<idx_b_bits>{lineaddr};
        index_m = val<idx_m_bits>{lineaddr};

        index_g.fanout(hard<LINEINST>{});
        index_b.fanout(hard<LINEINST>{});
        index_m.fanout(hard<LINEINST>{});

        for (u64 i=0; i<LINEINST; i++) {
            g_hi[i] = table_g_hi[i].read(index_g);
            b_hi[i] = table_b_hi[i].read(index_b);
            m_hi[i] = table_m_hi[i].read(index_m);
        }

        p_g = g_hi.concat();
        p_b = b_hi.concat();
        p_m = m_hi.concat();

        // p_b used by predict1 (1), reuse_predict1 (up to LINEINST-1),
        // predict2 (1 to build p_final), update_cycle (1 for b_err).
        p_b.fanout(hard<LINEINST+2>{});
        // p_g used in predict2 (1) + update_cycle (1).
        p_g.fanout(hard<2>{});
        // p_m used in predict2 (1 to build pm) + update_cycle (1).
        p_m.fanout(hard<2>{});

        return (block_entry & p_b) != hard<0>{};
    }

    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc)
    {
        return ((block_entry << block_size) & p_b) != hard<0>{};
    }

    val<1> predict2([[maybe_unused]] val<64> inst_pc)
    {
        // Build tournament prediction: meta=1 -> gshare, meta=0 -> bimodal.
        val<LINEINST> pm = p_m;
        pm.fanout(hard<2>{});
        p_final = (p_g & pm) | (p_b & ~pm);
        p_final.fanout(hard<LINEINST>{});
        val<1> taken = (block_entry & p_final) != hard<0>{};
        reuse_prediction(~val<1>{block_entry >> (LINEINST-1)});
        return taken;
    }

    val<1> reuse_predict2([[maybe_unused]] val<64> inst_pc)
    {
        val<1> taken = ((block_entry << block_size) & p_final) != hard<0>{};
        reuse_prediction(~val<1>{block_entry >> (LINEINST-1-block_size)});
        block_size++;
        return taken;
    }

    void update_condbr(val<64> branch_pc, val<1> taken, [[maybe_unused]] val<64> next_pc)
    {
        assert(num_branch < LINEINST);
        branch_offset[num_branch] = branch_pc >> 2;
        branch_dir[num_branch] = taken;
        num_branch++;
    }

    void update_cycle(instruction_info &block_end_info)
    {
        val<1> &mispredict = block_end_info.is_mispredict;
        val<64> &next_pc = block_end_info.next_pc;

        if (num_branch == 0) {
            // no conditional branch in this block
            val<1> line_end = block_entry >> (LINEINST - block_size);
            execute_if(~true_block | ~line_end, [&](){
                global_history = (global_history << 1) ^ val<GHIST>{next_pc >> 2};
                true_block = 1;
            });
            return;
        }

        mispredict.fanout(hard<2>{});
        branch_offset.fanout(hard<LINEINST>{});
        branch_dir.fanout(hard<3>{});
        index_g.fanout(hard<LINEINST*2>{});
        index_b.fanout(hard<LINEINST*2>{});
        index_m.fanout(hard<LINEINST*2>{});

        // Per-offset mask telling which update slot (if any) targets this offset.
        u64 update_valid = (u64(1)<<num_branch) - 1;
        arr<val<LINEINST>,LINEINST> update_mask = [&](u64 offset){
            arr<val<1>,LINEINST> match_offset = [&](u64 i){ return branch_offset[i] == offset; };
            return match_offset.concat() & update_valid;
        };
        update_mask.fanout(hard<2>{});

        arr<val<1>,LINEINST> is_branch = [&](u64 offset){
            return update_mask[offset] != hard<0>{};
        };
        is_branch.fanout(hard<4>{});

        val<LINEINST> branch_mask = is_branch.concat();
        branch_mask.fanout(hard<3>{});

        val<LINEINST> actualdirs = branch_dir.concat();
        actualdirs.fanout(hard<LINEINST>{});

        arr<val<1>,LINEINST> branch_taken = [&](u64 offset){
            return (actualdirs & update_mask[offset]) != hard<0>{};
        };
        branch_taken.fanout(hard<3>{});

        // Offset-indexed vector of actual branch directions (0 if no branch at offset).
        val<LINEINST> actual_at_offset = branch_taken.concat();
        actual_at_offset.fanout(hard<2>{});

        // Per-branch correctness vectors for the two component predictors.
        val<LINEINST> g_err_vec = (p_g ^ actual_at_offset) & branch_mask;
        val<LINEINST> b_err_vec = (p_b ^ actual_at_offset) & branch_mask;
        g_err_vec.fanout(hard<3>{});
        b_err_vec.fanout(hard<3>{});

        // Meta update happens only when exactly one component was correct.
        val<LINEINST> m_update_vec = (g_err_vec ^ b_err_vec) & branch_mask;
        m_update_vec.fanout(hard<4>{});
        // m_target = 1 means "gshare was correct" -> meta should select gshare.
        val<LINEINST> m_target_vec = (~g_err_vec) & m_update_vec;
        m_target_vec.fanout(hard<2>{});
        // Meta "err" = meta output disagreed with target at positions needing update.
        val<LINEINST> m_err_vec = (p_m ^ m_target_vec) & m_update_vec;

        arr<val<1>,LINEINST> g_err = g_err_vec.make_array(val<1>{});
        arr<val<1>,LINEINST> b_err = b_err_vec.make_array(val<1>{});
        arr<val<1>,LINEINST> m_upd = m_update_vec.make_array(val<1>{});
        arr<val<1>,LINEINST> m_tgt = m_target_vec.make_array(val<1>{});
        arr<val<1>,LINEINST> m_err = m_err_vec.make_array(val<1>{});

        g_err.fanout(hard<2>{});
        b_err.fanout(hard<2>{});
        m_err.fanout(hard<2>{});

        // Read hysteresis bits where a flip might be warranted.
        arr<val<1>,LINEINST> g_weak = [&](u64 i) -> val<1> {
            return execute_if(g_err[i], [&](){ return ~table_g_lo[i].read(index_g); });
        };
        arr<val<1>,LINEINST> b_weak = [&](u64 i) -> val<1> {
            return execute_if(b_err[i], [&](){ return ~table_b_lo[i].read(index_b); });
        };
        arr<val<1>,LINEINST> m_weak = [&](u64 i) -> val<1> {
            return execute_if(m_err[i], [&](){ return ~table_m_lo[i].read(index_m); });
        };

        // Extra cycle whenever any component's hysteresis read + write collision
        // can occur. Our reads of _lo tables are gated by g_err / b_err / m_err,
        // so the cycle only needs to advance when at least one of those is true.
        // g_err XOR b_err = m_update; g_err AND b_err implies mispredict (the
        // last branch, since block would have ended earlier otherwise).
        need_extra_cycle(mispredict | (m_update_vec != hard<0>{}));

        for (u64 i=0; i<LINEINST; i++) {
            // Gshare: flip prediction bit if wrong and hysteresis is weak; always
            // refresh hysteresis (strong=correct, weak=wrong) when the offset saw a branch.
            execute_if(g_weak[i], [&](){ table_g_hi[i].write(index_g, branch_taken[i]); });
            execute_if(is_branch[i], [&](){ table_g_lo[i].write(index_g, ~g_err[i]); });

            // Bimodal: same pattern.
            execute_if(b_weak[i], [&](){ table_b_hi[i].write(index_b, branch_taken[i]); });
            execute_if(is_branch[i], [&](){ table_b_lo[i].write(index_b, ~b_err[i]); });

            // Meta: flip meta when components disagreed on correctness and hysteresis is weak;
            // refresh meta hysteresis whenever components disagreed on correctness.
            execute_if(m_weak[i], [&](){ table_m_hi[i].write(index_m, m_tgt[i]); });
            execute_if(m_upd[i], [&](){ table_m_lo[i].write(index_m, ~m_err[i]); });
        }

        // Global history update (same "true block" rule as gshare.hpp).
        val<1> line_end = block_entry >> (LINEINST - block_size);
        true_block = ~mispredict | branch_dir[num_branch-1] | line_end;
        execute_if(true_block, [&](){
            global_history = (global_history << 1) ^ val<GHIST>{next_pc >> 2};
        });

        num_branch = 0;
    }
};
