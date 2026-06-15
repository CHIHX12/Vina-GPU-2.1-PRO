#pragma once
// LS metal-coordination rescoring (Part A). Extracted from main_procedure_cl.cpp.
#include <vector>
#include <array>
#include "common.h"
#include "model.h"
#include "conf.h"

bool is_receptor_metal(sz ad_type);
std::vector<std::array<float,3>> collect_receptor_metals(const model& m);
void apply_ls_metal_scores(std::vector<output_type>& poses,
                           const std::vector<std::array<float,3>>& metals, float weight);
