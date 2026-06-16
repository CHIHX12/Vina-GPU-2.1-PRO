#pragma once
// Build a GPU m_cl from a Vina model: atom types/coords/charges, intra-ligand pairs, the rigid-body
// forward-kinematics tree, the flexible side-chain forest (empty for rigid/no-flex), and the
// ligand<->flex other_pairs. The caller mallocs m_ptr. Extracted from main_procedure_cl.cpp.
#include "model.h"
#include "kernel2.h"   // m_cl + MAX_* bounds macros

void fill_m_cl(model& m, m_cl* m_ptr);

// Native multi-ligand (P1): build one m_multi_cl from a model that already holds N ligands
// (constructed host-side via parse_bundle()/model::append()). Each ligand becomes an independent
// rigid root with its own tree and intra-ligand pairs; inter-ligand pairs come from
// m.get_other_pairs() (model::append() fills them). Ligand-only for P1 (throws if flex present).
void fill_m_cl_multi(model& m, m_multi_cl* m_ptr);
