#pragma once
// Build a GPU m_cl from a Vina model: atom types/coords/charges, intra-ligand pairs, the rigid-body
// forward-kinematics tree, the flexible side-chain forest (empty for rigid/no-flex), and the
// ligand<->flex other_pairs. The caller mallocs m_ptr. Extracted from main_procedure_cl.cpp.
#include "model.h"
#include "kernel2.h"   // m_cl + MAX_* bounds macros

void fill_m_cl(model& m, m_cl* m_ptr);
