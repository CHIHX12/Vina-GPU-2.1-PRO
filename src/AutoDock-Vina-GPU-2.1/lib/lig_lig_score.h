#pragma once
#include "model.h"
#include "precalculate.h"

// Compute Vina pairwise interaction energy between movable atoms of ma and mb.
// Both models must have current coordinates set (m.coords must be up to date).
// Mirrors the non_cache::eval() pattern but operates across two separate models.
fl eval_lig_lig(const model& ma, const model& mb, const precalculate& p);
