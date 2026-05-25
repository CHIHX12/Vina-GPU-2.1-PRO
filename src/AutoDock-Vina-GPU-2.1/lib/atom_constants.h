/*

   Copyright (c) 2006-2010, The Scripps Research Institute

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Author: Dr. Oleg Trott <ot14@columbia.edu>, 
           The Olson Lab, 
           The Scripps Research Institute

*/

#ifndef VINA_ATOM_CONSTANTS_H
#define VINA_ATOM_CONSTANTS_H

#include "common.h"

// based on SY_TYPE_* but includes H
const sz EL_TYPE_H    =  0;
const sz EL_TYPE_C    =  1;
const sz EL_TYPE_N    =  2;
const sz EL_TYPE_O    =  3;
const sz EL_TYPE_S    =  4;
const sz EL_TYPE_P    =  5;
const sz EL_TYPE_F    =  6;
const sz EL_TYPE_Cl   =  7;
const sz EL_TYPE_Br   =  8;
const sz EL_TYPE_I    =  9;
const sz EL_TYPE_Met  = 10;
const sz EL_TYPE_SIZE = 11;

// AutoDock4
const sz AD_TYPE_C    =  0;
const sz AD_TYPE_A    =  1;
const sz AD_TYPE_N    =  2;
const sz AD_TYPE_O    =  3;
const sz AD_TYPE_P    =  4;
const sz AD_TYPE_S    =  5;
const sz AD_TYPE_H    =  6; // non-polar hydrogen
const sz AD_TYPE_F    =  7;
const sz AD_TYPE_I    =  8;
const sz AD_TYPE_NA   =  9;
const sz AD_TYPE_OA   = 10;
const sz AD_TYPE_SA   = 11;
const sz AD_TYPE_HD   = 12;
const sz AD_TYPE_Mg   = 13;
const sz AD_TYPE_Mn   = 14;
const sz AD_TYPE_Zn   = 15;
const sz AD_TYPE_Ca   = 16;
const sz AD_TYPE_Fe   = 17;
const sz AD_TYPE_Cl   = 18;
const sz AD_TYPE_Br   = 19;
// Extended metal types (20-31) — CSD-based coordination parameters in kernel1.cl
const sz AD_TYPE_Mo   = 20; // Molybdenum  (nitrogenase, xanthine oxidase)
const sz AD_TYPE_W    = 21; // Tungsten    (W-enzymes in thermophiles)
const sz AD_TYPE_Cu   = 22; // Copper      (SOD, plastocyanin, laccase)
const sz AD_TYPE_Co   = 23; // Cobalt      (B12 cobalamin, nitrile hydratase)
const sz AD_TYPE_Ni   = 24; // Nickel      (urease, hydrogenase, CO dehydrogenase)
const sz AD_TYPE_V    = 25; // Vanadium    (haloperoxidase, nitrogenase alternative)
const sz AD_TYPE_Pt   = 26; // Platinum    (cisplatin/oxaliplatin drug targets)
const sz AD_TYPE_Pd   = 27; // Palladium   (metallodrug targets)
const sz AD_TYPE_Ru   = 28; // Ruthenium   (anticancer metallodrugs)
const sz AD_TYPE_Rh   = 29; // Rhodium     (metallodrug targets)
const sz AD_TYPE_Cd   = 30; // Cadmium     (metallothionein, toxicology)
const sz AD_TYPE_Hg   = 31; // Mercury     (toxicology, resistance proteins)
const sz AD_TYPE_SIZE = 32;

// X-Score
const sz XS_TYPE_C_H   =  0;
const sz XS_TYPE_C_P   =  1;
const sz XS_TYPE_N_P   =  2;
const sz XS_TYPE_N_D   =  3;
const sz XS_TYPE_N_A   =  4;
const sz XS_TYPE_N_DA  =  5;
const sz XS_TYPE_O_P   =  6;
const sz XS_TYPE_O_D   =  7;
const sz XS_TYPE_O_A   =  8;
const sz XS_TYPE_O_DA  =  9;
const sz XS_TYPE_S_P   = 10;
const sz XS_TYPE_P_P   = 11;
const sz XS_TYPE_F_H   = 12;
const sz XS_TYPE_Cl_H  = 13;
const sz XS_TYPE_Br_H  = 14;
const sz XS_TYPE_I_H   = 15;
const sz XS_TYPE_Met_D = 16;
const sz XS_TYPE_SIZE  = 17;

// DrugScore-CSD
const sz SY_TYPE_C_3   =  0;
const sz SY_TYPE_C_2   =  1;
const sz SY_TYPE_C_ar  =  2;
const sz SY_TYPE_C_cat =  3;
const sz SY_TYPE_N_3   =  4;
const sz SY_TYPE_N_ar  =  5;
const sz SY_TYPE_N_am  =  6;
const sz SY_TYPE_N_pl3 =  7;
const sz SY_TYPE_O_3   =  8;
const sz SY_TYPE_O_2   =  9;
const sz SY_TYPE_O_co2 = 10;
const sz SY_TYPE_S     = 11;
const sz SY_TYPE_P     = 12;
const sz SY_TYPE_F     = 13;
const sz SY_TYPE_Cl    = 14;
const sz SY_TYPE_Br    = 15;
const sz SY_TYPE_I     = 16;
const sz SY_TYPE_Met   = 17;
const sz SY_TYPE_SIZE  = 18;

struct atom_kind {
    std::string name;
    fl radius;
	fl depth;
	fl solvation;
	fl volume;
	fl covalent_radius; // from http://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
};

// generated from edited AD4_parameters.data using a script, 
// then covalent radius added from en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
const atom_kind atom_kind_data[] = { // name, radius, depth, solvation parameter, volume, covalent radius
	{ "C",    2.00000,    0.15000,   -0.00143,   33.51030,   0.77}, //  0
	{ "A",    2.00000,    0.15000,   -0.00052,   33.51030,   0.77}, //  1
	{ "N",    1.75000,    0.16000,   -0.00162,   22.44930,   0.75}, //  2
	{ "O",    1.60000,    0.20000,   -0.00251,   17.15730,   0.73}, //  3
	{ "P",    2.10000,    0.20000,   -0.00110,   38.79240,   1.06}, //  4
	{ "S",    2.00000,    0.20000,   -0.00214,   33.51030,   1.02}, //  5
	{ "H",    1.00000,    0.02000,    0.00051,    0.00000,   0.37}, //  6
	{ "F",    1.54500,    0.08000,   -0.00110,   15.44800,   0.71}, //  7
	{ "I",    2.36000,    0.55000,   -0.00110,   55.05850,   1.33}, //  8
	{"NA",    1.75000,    0.16000,   -0.00162,   22.44930,   0.75}, //  9
	{"OA",    1.60000,    0.20000,   -0.00251,   17.15730,   0.73}, // 10
	{"SA",    2.00000,    0.20000,   -0.00214,   33.51030,   1.02}, // 11
	{"HD",    1.00000,    0.02000,    0.00051,    0.00000,   0.37}, // 12
	{"Mg",    0.65000,    0.87500,   -0.00110,    1.56000,   1.30}, // 13
	{"Mn",    0.65000,    0.87500,   -0.00110,    2.14000,   1.39}, // 14
	{"Zn",    0.74000,    0.55000,   -0.00110,    1.70000,   1.31}, // 15
	{"Ca",    0.99000,    0.55000,   -0.00110,    2.77000,   1.74}, // 16
	{"Fe",    0.65000,    0.01000,   -0.00110,    1.84000,   1.25}, // 17
	{"Cl",    2.04500,    0.27600,   -0.00110,   35.82350,   0.99}, // 18
	{"Br",    2.16500,    0.38900,   -0.00110,   42.56610,   1.14}, // 19
	// Extended metals — vdW radius/depth borrowed from closest AD4 analog;
	// coordination Gaussian parameters are in kernel1.cl (CSD-based)
	{"Mo",    0.65000,    0.87500,   -0.00110,    2.50000,   1.54}, // 20
	{"W",     0.65000,    0.87500,   -0.00110,    2.80000,   1.62}, // 21
	{"Cu",    0.70000,    0.55000,   -0.00110,    1.90000,   1.38}, // 22
	{"Co",    0.65000,    0.87500,   -0.00110,    1.84000,   1.32}, // 23
	{"Ni",    0.65000,    0.87500,   -0.00110,    1.84000,   1.25}, // 24
	{"V",     0.65000,    0.87500,   -0.00110,    2.14000,   1.35}, // 25
	{"Pt",    0.80000,    0.55000,   -0.00110,    2.50000,   1.36}, // 26
	{"Pd",    0.80000,    0.55000,   -0.00110,    2.30000,   1.31}, // 27
	{"Ru",    0.70000,    0.87500,   -0.00110,    2.20000,   1.34}, // 28
	{"Rh",    0.70000,    0.87500,   -0.00110,    2.10000,   1.35}, // 29
	{"Cd",    0.78000,    0.55000,   -0.00110,    2.20000,   1.48}, // 30
	{"Hg",    0.83000,    0.55000,   -0.00110,    2.40000,   1.49}, // 31
};

const fl metal_solvation_parameter = -0.00110;

const fl metal_covalent_radius = 1.75; // for metals not on the list // FIXME this info should be moved to non_ad_metals

const sz atom_kinds_size =  sizeof(atom_kind_data) / sizeof(const atom_kind);

struct atom_equivalence {
	std::string name;
	std::string to;
};

const atom_equivalence atom_equivalence_data[] = {
	{"Se",  "S" },
	// Boron: boronic acid warhead drugs (bortezomib, tavaborole, crisaborole) — treat as C-like
	{"B",   "C" },
	// Silicon: organosilicon compounds — treat as C-like
	{"Si",  "C" },
	{"SI",  "C" },
	// Uppercase aliases for metals unknown to standard AutoDockTools:
	// PDB element column is all-uppercase ("MO","PT",...); these map to our mixed-case entries.
	{"MO",  "Mo"},
	{"PT",  "Pt"},
	{"PD",  "Pd"},
	{"RU",  "Ru"},
	{"RH",  "Rh"},
	// Co/Ni/Cu/Cd/Hg confirmed mixed-case from AutoDockTools (were in non_ad_metal_names),
	// but add uppercase aliases as safety net for non-ADT prepared files.
	{"CU",  "Cu"},
	{"CO",  "Co"},
	{"NI",  "Ni"},
	{"CD",  "Cd"},
	{"HG",  "Hg"},
	// Arsenic uppercase alias (PDB element column uses "AS")
	{"AS",  "As"},
	// Previously-added rare metals — uppercase PDB aliases
	{"SB",  "Sb"},  // Antimony
	{"BI",  "Bi"},  // Bismuth
	{"LI",  "Li"},  // Lithium
	{"AL",  "Al"},  // Aluminum
	{"SN",  "Sn"},  // Tin
	{"TE",  "Te"},  // Tellurium
	{"GE",  "Ge"},  // Germanium
	// ── Complete periodic-table metal coverage (uppercase PDB → mixed-case) ──
	// Alkali metals
	{"RB",  "Rb"},  // Rubidium  (Rb-82 PET cardiac imaging)
	{"CS",  "Cs"},  // Cesium    (Cs-131 brachytherapy)
	// Alkaline earth
	{"BE",  "Be"},  // Beryllium (rare, berylliosis research)
	{"SR",  "Sr"},  // Strontium (strontium ranelate, Sr-89 therapy)
	{"BA",  "Ba"},  // Barium    (contrast agents)
	{"RA",  "Ra"},  // Radium    (Ra-223 Xofigo, bone metastases)
	// 4th-period transition metals
	{"SC",  "Sc"},  // Scandium  (Sc-47 radiotherapy)
	{"TI",  "Ti"},  // Titanium  (titanocene dichloride anticancer)
	{"CR",  "Cr"},  // Chromium  (chromium picolinate, coordination complexes)
	// 5th-period transition metals
	{"Y",   "Y" },  // Yttrium   (Y-90 microspheres, liver cancer) — single-char, PDB: " Y"
	{"ZR",  "Zr"},  // Zirconium (Zr-89 PET, zirconocene antitumor)
	{"NB",  "Nb"},  // Niobium   (niobocene anticancer research)
	{"TC",  "Tc"},  // Technetium(Tc-99m — most widely used radiopharmaceutical)
	{"AG",  "Ag"},  // Silver    (silver sulfadiazine, antimicrobial)
	// 6th-period transition metals
	{"HF",  "Hf"},  // Hafnium   (hafnocene anticancer, Hf-178 isomers)
	{"TA",  "Ta"},  // Tantalum  (Ta implants, rare drug targets)
	{"RE",  "Re"},  // Rhenium   (Re-186/188 radiotherapy, fac-[Re(CO)3])
	{"OS",  "Os"},  // Osmium    (NAMI-A analogs, anticancer)
	{"IR",  "Ir"},  // Iridium   (RAPTA-C analogs, photodynamic anticancer)
	{"AU",  "Au"},  // Gold      (auranofin RA drug, gold thiolates)
	// Post-transition metals
	{"GA",  "Ga"},  // Gallium   (gallium nitrate, Ga-68 PET imaging)
	{"IN",  "In"},  // Indium    (In-111 Octreoscan, indium-based drugs)
	{"TL",  "Tl"},  // Thallium  (Tl-201 cardiac imaging)
	{"PB",  "Pb"},  // Lead      (Pb-212 TAT, historical Pb compounds)
	// Lanthanides (rare earth — important for MRI contrast & radiotherapy)
	{"LA",  "La"},  // Lanthanum  (lanthanum carbonate, phosphate binder)
	{"CE",  "Ce"},  // Cerium     (cerium nitrate, wound care)
	{"PR",  "Pr"},  // Praseodymium
	{"ND",  "Nd"},  // Neodymium
	{"PM",  "Pm"},  // Promethium (Pm-147 radiotherapy)
	{"SM",  "Sm"},  // Samarium   (Sm-153 EDTMP bone pain palliation)
	{"EU",  "Eu"},  // Europium   (luminescent probes)
	{"GD",  "Gd"},  // Gadolinium (MRI contrast agents: Magnevist etc.)
	{"TB",  "Tb"},  // Terbium    (Tb-149/161 radiotherapy)
	{"DY",  "Dy"},  // Dysprosium (Dy-165 synovectomy)
	{"HO",  "Ho"},  // Holmium    (Ho-166 radiotherapy)
	{"ER",  "Er"},  // Erbium     (Er-169 radiotherapy)
	{"TM",  "Tm"},  // Thulium    (Tm-170 brachytherapy)
	{"YB",  "Yb"},  // Ytterbium  (Yb-175 radiotherapy, CT contrast)
	{"LU",  "Lu"},  // Lutetium   (Lu-177 DOTATATE — approved cancer therapy)
	// Actinides
	{"AC",  "Ac"},  // Actinium   (Ac-225 targeted alpha therapy)
	{"TH",  "Th"},  // Thorium    (Th-227 radiotherapy)
};

const sz atom_equivalences_size = sizeof(atom_equivalence_data) / sizeof(const atom_equivalence);

struct acceptor_kind {
	sz ad_type;
	fl radius;
	fl depth;
};

const acceptor_kind acceptor_kind_data[] = { // ad_type, optimal length, depth
	{AD_TYPE_NA, 1.9, 5.0},
	{AD_TYPE_OA, 1.9, 5.0},
	{AD_TYPE_SA, 2.5, 1.0}
};

const sz acceptor_kinds_size = sizeof(acceptor_kind_data) / sizeof(acceptor_kind);

inline bool ad_is_hydrogen(sz ad) {
	return ad == AD_TYPE_H || ad == AD_TYPE_HD;
}

inline bool ad_is_heteroatom(sz ad) { // returns false for ad >= AD_TYPE_SIZE
	return ad != AD_TYPE_A && ad != AD_TYPE_C  && 
		   ad != AD_TYPE_H && ad != AD_TYPE_HD && 
		   ad < AD_TYPE_SIZE;
}

inline sz ad_type_to_el_type(sz t) {
	switch(t) {
		case AD_TYPE_C    : return EL_TYPE_C;
		case AD_TYPE_A    : return EL_TYPE_C;
		case AD_TYPE_N    : return EL_TYPE_N;
		case AD_TYPE_O    : return EL_TYPE_O;
		case AD_TYPE_P    : return EL_TYPE_P;
		case AD_TYPE_S    : return EL_TYPE_S;
		case AD_TYPE_H    : return EL_TYPE_H;
		case AD_TYPE_F    : return EL_TYPE_F;
		case AD_TYPE_I    : return EL_TYPE_I;
		case AD_TYPE_NA   : return EL_TYPE_N;
		case AD_TYPE_OA   : return EL_TYPE_O;
		case AD_TYPE_SA   : return EL_TYPE_S;
		case AD_TYPE_HD   : return EL_TYPE_H;
		case AD_TYPE_Mg   : return EL_TYPE_Met;
		case AD_TYPE_Mn   : return EL_TYPE_Met;
		case AD_TYPE_Zn   : return EL_TYPE_Met;
		case AD_TYPE_Ca   : return EL_TYPE_Met;
		case AD_TYPE_Fe   : return EL_TYPE_Met;
		case AD_TYPE_Cl   : return EL_TYPE_Cl;
		case AD_TYPE_Br   : return EL_TYPE_Br;
		case AD_TYPE_Mo   : return EL_TYPE_Met;
		case AD_TYPE_W    : return EL_TYPE_Met;
		case AD_TYPE_Cu   : return EL_TYPE_Met;
		case AD_TYPE_Co   : return EL_TYPE_Met;
		case AD_TYPE_Ni   : return EL_TYPE_Met;
		case AD_TYPE_V    : return EL_TYPE_Met;
		case AD_TYPE_Pt   : return EL_TYPE_Met;
		case AD_TYPE_Pd   : return EL_TYPE_Met;
		case AD_TYPE_Ru   : return EL_TYPE_Met;
		case AD_TYPE_Rh   : return EL_TYPE_Met;
		case AD_TYPE_Cd   : return EL_TYPE_Met;
		case AD_TYPE_Hg   : return EL_TYPE_Met;
		case AD_TYPE_SIZE : return EL_TYPE_SIZE;
		default: VINA_CHECK(false);
	}
	return EL_TYPE_SIZE; // to placate the compiler in case of warnings - it should never get here though
}

const fl xs_vdw_radii[] = {
	1.9, // C_H
	1.9, // C_P
	1.8, // N_P
	1.8, // N_D
	1.8, // N_A
	1.8, // N_DA
	1.7, // O_P
	1.7, // O_D
	1.7, // O_A
	1.7, // O_DA
	2.0, // S_P
	2.1, // P_P
	1.5, // F_H
	1.8, // Cl_H
	2.0, // Br_H
	2.2, // I_H
	1.2  // Met_D
};


inline fl xs_radius(sz t) {
	const sz n = sizeof(xs_vdw_radii) / sizeof(const fl);
	assert(n == XS_TYPE_SIZE);
	assert(t < n);
	return xs_vdw_radii[t];
}

const std::string non_ad_metal_names[] = { // metals/metalloids not in atom_kind_data — get XS_TYPE_Met_D
	// Note: Cu, Co, Ni, Hg, Cd removed — now proper AD types (20-31)
	// Fe kept for compatibility (also in atom_kind_data at 17)
	// As (Arsenic): organoarsenic drug ligands (e.g. 4BKS/CAS melarsoprol)
	// Sb (Antimony): antimonial antiparasitic drugs (meglumine antimoniate)
	// Bi (Bismuth): bismuth-based drugs
	// Li (Lithium): psychiatric ligands
	// Al (Aluminum): antacid/adjuvant ligands
	// Sn (Tin): organotin compounds
	// Te (Tellurium): rare chalcogenide drugs
	// Ge (Germanium): rare organogermanium compounds
	// Previously-in-list
	"Fe", "Na", "K", "U",
	// Metalloids
	"As", "Sb", "Bi", "Te", "Ge",
	// Previously-added rare metals
	"Li", "Al", "Sn",
	// ── Complete periodic-table coverage ──────────────────────────────────────
	// Alkali metals
	"Rb", "Cs",
	// Alkaline earth
	"Be", "Sr", "Ba", "Ra",
	// 4th-period transition
	"Sc", "Ti", "Cr",
	// 5th-period transition
	"Y",  "Zr", "Nb", "Tc", "Ag",
	// 6th-period transition
	"Hf", "Ta", "Re", "Os", "Ir", "Au",
	// Post-transition metals
	"Ga", "In", "Tl", "Pb",
	// Lanthanides
	"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu",
	"Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
	// Actinides
	"Ac", "Th"
};

inline bool is_non_ad_metal_name(const std::string& name) {
	const sz s = sizeof(non_ad_metal_names) / sizeof(const std::string);
	VINA_FOR(i, s)
		if(non_ad_metal_names[i] == name)
			return true;
	return false;
}

inline bool xs_is_hydrophobic(sz xs) {
	return xs == XS_TYPE_C_H || 
		   xs == XS_TYPE_F_H ||
		   xs == XS_TYPE_Cl_H ||
		   xs == XS_TYPE_Br_H || 
		   xs == XS_TYPE_I_H;
}

inline bool xs_is_acceptor(sz xs) {
	return xs == XS_TYPE_N_A ||
		   xs == XS_TYPE_N_DA ||
		   xs == XS_TYPE_O_A ||
		   xs == XS_TYPE_O_DA;
}

inline bool xs_is_donor(sz xs) {
	return xs == XS_TYPE_N_D ||
		   xs == XS_TYPE_N_DA ||
		   xs == XS_TYPE_O_D ||
		   xs == XS_TYPE_O_DA ||
		   xs == XS_TYPE_Met_D;
}

inline bool xs_donor_acceptor(sz t1, sz t2) {
	return xs_is_donor(t1) && xs_is_acceptor(t2);
}

inline bool xs_h_bond_possible(sz t1, sz t2) {
	return xs_donor_acceptor(t1, t2) || xs_donor_acceptor(t2, t1);
}

inline const atom_kind& ad_type_property(sz i) {
	assert(AD_TYPE_SIZE == atom_kinds_size);
    assert(i < atom_kinds_size);
    return atom_kind_data[i];
}

inline sz string_to_ad_type(const std::string& name) { // returns AD_TYPE_SIZE if not found (no exceptions thrown, because metals unknown to AD4 are not exceptional)
    VINA_FOR(i, atom_kinds_size)
		if(atom_kind_data[i].name == name)
			return i;
	VINA_FOR(i, atom_equivalences_size)
		if(atom_equivalence_data[i].name == name)
			return string_to_ad_type(atom_equivalence_data[i].to);
    return AD_TYPE_SIZE;
}

inline fl max_covalent_radius() {
	fl tmp = 0;
	VINA_FOR(i, atom_kinds_size)
		if(atom_kind_data[i].covalent_radius > tmp)
			tmp = atom_kind_data[i].covalent_radius;
	return tmp;
}

#endif
