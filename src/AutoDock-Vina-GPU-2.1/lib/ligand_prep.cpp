#include "ligand_prep.h"
#include <stdexcept>
#include <string>

using namespace std;

void fill_m_cl(model& m, m_cl* m_ptr) {
	m_ptr->flex_rigid.num_nodes = 0;   // Stage 1: flex forest empty until populated
	m_ptr->other_pairs.num_pairs = 0;  // Stage 2: ligand↔flex pairs empty until populated
	for (int ai = 0; ai < (int)m.atoms.size(); ai++) {
		m_ptr->atoms[ai].types[0] = m.atoms[ai].el;
		m_ptr->atoms[ai].types[1] = m.atoms[ai].ad;
		m_ptr->atoms[ai].types[2] = m.atoms[ai].xs;
		m_ptr->atoms[ai].types[3] = m.atoms[ai].sy;
		for (int j = 0; j < 3; j++) m_ptr->atoms[ai].coords[j] = m.atoms[ai].coords[j];
		m_ptr->atoms[ai].charge = (float)m.atoms[ai].charge; // QFD: partial charge [e]
	}
	for (int ci = 0; ci < (int)m.coords.size(); ci++)
		for (int j = 0; j < 3; j++) m_ptr->m_coords.coords[ci][j] = m.coords[ci].data[j];
	for (int ci = 0; ci < (int)m.coords.size(); ci++)
		for (int j = 0; j < 3; j++) m_ptr->minus_forces.coords[ci][j] = m.minus_forces[ci].data[j];


	ligand m_ligand = m.ligands[0];
	if (m_ligand.end >= MAX_NUM_OF_ATOMS) { throw std::runtime_error("Ligand too large! The maximum number of atoms is " +
	 std::to_string(MAX_NUM_OF_ATOMS) + " and the ligand has " + std::to_string(m_ligand.end) + " atoms."); }
	m_ptr->ligand.pairs.num_pairs = m.ligands[0].pairs.size();
	if (m.ligands[0].pairs.size() >= MAX_NUM_OF_LIG_PAIRS) { throw std::runtime_error("Ligand too large! The maximum number of pairs is " +
	 std::to_string(MAX_NUM_OF_LIG_PAIRS) + " and the ligand has " + std::to_string(m.ligands[0].pairs.size()) + " pairs."); }
	for (int pi = 0; pi < m_ptr->ligand.pairs.num_pairs; pi++) {
		m_ptr->ligand.pairs.type_pair_index[pi] = m.ligands[0].pairs[pi].type_pair_index;
		m_ptr->ligand.pairs.a[pi] = m.ligands[0].pairs[pi].a;
		m_ptr->ligand.pairs.b[pi] = m.ligands[0].pairs[pi].b;
	}
	m_ptr->ligand.begin = m.ligands[0].begin;
	m_ptr->ligand.end   = m.ligands[0].end;
	m_ptr->ligand.rigid.atom_range[0][0] = m_ligand.node.begin;
	m_ptr->ligand.rigid.atom_range[0][1] = m_ligand.node.end;
	for (int j = 0; j < 3; j++) m_ptr->ligand.rigid.origin[0][j] = m_ligand.node.get_origin()[j];
	for (int j = 0; j < 9; j++) m_ptr->ligand.rigid.orientation_m[0][j] = m_ligand.node.get_orientation_m().data[j];
	m_ptr->ligand.rigid.orientation_q[0][0] = m_ligand.node.orientation().R_component_1();
	m_ptr->ligand.rigid.orientation_q[0][1] = m_ligand.node.orientation().R_component_2();
	m_ptr->ligand.rigid.orientation_q[0][2] = m_ligand.node.orientation().R_component_3();
	m_ptr->ligand.rigid.orientation_q[0][3] = m_ligand.node.orientation().R_component_4();
	for (int j = 0; j < 3; j++) { m_ptr->ligand.rigid.axis[0][j] = 0; m_ptr->ligand.rigid.relative_axis[0][j] = 0; m_ptr->ligand.rigid.relative_origin[0][j] = 0; }
	struct tmp_struct_local {
		int start_index = 0;
		int parent_index = 0;
		void store_node(tree<segment>& child_ptr, rigid_cl& rigid) {
			start_index++;
			rigid.parent[start_index] = parent_index;
			rigid.atom_range[start_index][0] = child_ptr.node.begin;
			rigid.atom_range[start_index][1] = child_ptr.node.end;
			for (int j = 0; j < 9; j++) rigid.orientation_m[start_index][j] = child_ptr.node.get_orientation_m().data[j];
			rigid.orientation_q[start_index][0] = child_ptr.node.orientation().R_component_1();
			rigid.orientation_q[start_index][1] = child_ptr.node.orientation().R_component_2();
			rigid.orientation_q[start_index][2] = child_ptr.node.orientation().R_component_3();
			rigid.orientation_q[start_index][3] = child_ptr.node.orientation().R_component_4();
			for (int j = 0; j < 3; j++) {
				rigid.origin[start_index][j]          = child_ptr.node.get_origin()[j];
				rigid.axis[start_index][j]            = child_ptr.node.get_axis()[j];
				rigid.relative_axis[start_index][j]   = child_ptr.node.relative_axis[j];
				rigid.relative_origin[start_index][j] = child_ptr.node.relative_origin[j];
			}
			if (child_ptr.children.empty()) return;
			if (start_index >= MAX_NUM_OF_RIGID) { throw std::runtime_error("Children map too large!"); }
			int parent_index_tmp = start_index;
			for (int ci = 0; ci < (int)child_ptr.children.size(); ci++) {
				this->parent_index = parent_index_tmp;
				this->store_node(child_ptr.children[ci], rigid);
			}
		}
	};
	tmp_struct_local ts;
	for (int ci = 0; ci < (int)m_ligand.children.size(); ci++) {
		ts.parent_index = 0;
		ts.store_node(m_ligand.children[ci], m_ptr->ligand.rigid);
	}
	m_ptr->ligand.rigid.num_children = ts.start_index;
	m_ptr->ligand.rigid.parent[0] = -1;  // root has no parent (children derived from parent[])

	// ───── Stage 1: populate flexible side-chain forest from m.flex ─────
	// Each residue (heterotree<first_segment>) → one root node (parent=-1,
	// first_segment, consumes residue torsions[0]) + DFS child segments.
	// torsion_idx counts flex torsions in conf-flattening order (residue, then DFS).
	{
		flex_rigid_cl& fr = m_ptr->flex_rigid;
		struct flex_store {
			void store_seg(tree<segment>& t, int parent, int& ni, int& ft, flex_rigid_cl& fr) {
				int me = ni++;
				if (me >= MAX_NUM_OF_FLEX_RIGID) throw std::runtime_error("Too many flex nodes (raise MAX_NUM_OF_FLEX_RIGID)!");
				fr.parent[me] = parent;
				fr.torsion_idx[me] = ft++;
				fr.atom_range[me][0] = t.node.begin;
				fr.atom_range[me][1] = t.node.end;
				for (int j = 0; j < 9; j++) fr.orientation_m[me][j] = t.node.get_orientation_m().data[j];
				fr.orientation_q[me][0] = t.node.orientation().R_component_1();
				fr.orientation_q[me][1] = t.node.orientation().R_component_2();
				fr.orientation_q[me][2] = t.node.orientation().R_component_3();
				fr.orientation_q[me][3] = t.node.orientation().R_component_4();
				for (int j = 0; j < 3; j++) {
					fr.origin[me][j]          = t.node.get_origin()[j];
					fr.axis[me][j]            = t.node.get_axis()[j];
					fr.relative_axis[me][j]   = t.node.relative_axis[j];
					fr.relative_origin[me][j] = t.node.relative_origin[j];
				}
				for (int ci = 0; ci < (int)t.children.size(); ci++)
					store_seg(t.children[ci], me, ni, ft, fr);
			}
		};
		flex_store fs;
		int ni = 0, ft = 0;
		vector_mutable<residue>& flex_res = m.get_flex();
		for (int ri = 0; ri < (int)flex_res.size(); ri++) {
			residue& res = flex_res[ri];
			int me = ni++;
			if (me >= MAX_NUM_OF_FLEX_RIGID) throw std::runtime_error("Too many flex nodes (raise MAX_NUM_OF_FLEX_RIGID)!");
			fr.parent[me]      = -1;        // first_segment root (rotates about its fixed axis)
			fr.torsion_idx[me] = ft++;      // residue torsions[0]
			fr.atom_range[me][0] = res.node.begin;
			fr.atom_range[me][1] = res.node.end;
			for (int j = 0; j < 9; j++) fr.orientation_m[me][j] = res.node.get_orientation_m().data[j];
			fr.orientation_q[me][0] = res.node.orientation().R_component_1();
			fr.orientation_q[me][1] = res.node.orientation().R_component_2();
			fr.orientation_q[me][2] = res.node.orientation().R_component_3();
			fr.orientation_q[me][3] = res.node.orientation().R_component_4();
			for (int j = 0; j < 3; j++) {
				fr.origin[me][j]          = res.node.get_origin()[j];
				fr.axis[me][j]            = res.node.get_axis()[j];
				fr.relative_axis[me][j]   = 0.0f;   // root: unused (no parent)
				fr.relative_origin[me][j] = 0.0f;
			}
			for (int ci = 0; ci < (int)res.children.size(); ci++)
				fs.store_seg(res.children[ci], me, ni, ft, fr);
		}
		fr.num_nodes = ni;
		if (ft > MAX_NUM_OF_FLEX_TORSION) throw std::runtime_error("Too many flex torsions (raise MAX_NUM_OF_FLEX_TORSION)!");
	}

	// ───── Stage 2: populate other_pairs (ligand↔flex + flex↔flex) ─────
	{
		const interacting_pairs& op = m.get_other_pairs();
		if ((int)op.size() >= MAX_NUM_OF_OTHER_PAIRS) throw std::runtime_error("Too many other_pairs (raise MAX_NUM_OF_OTHER_PAIRS)!");
		m_ptr->other_pairs.num_pairs = (int)op.size();
		for (int pi = 0; pi < (int)op.size(); pi++) {
			m_ptr->other_pairs.type_pair_index[pi] = (int)op[pi].type_pair_index;
			m_ptr->other_pairs.a[pi] = (int)op[pi].a;
			m_ptr->other_pairs.b[pi] = (int)op[pi].b;
		}
	}

	m_ptr->m_num_movable_atoms = m.num_movable_atoms();
}

// ════════════════ Native multi-ligand (P1) host fill ════════════════
// Maps a model holding N ligands (built via parse_bundle()/model::append()) into one m_multi_cl.
// Mirrors fill_m_cl() but loops over m.ligands[k]: each ligand gets its own rigid tree, its own
// slice of the shared lig_torsion[] (via torsion_offset), and its own intra-ligand pair list.
// Inter-ligand pairs are already in m.get_other_pairs() (model::append() builds them).
void fill_m_cl_multi(model& m, m_multi_cl* m_ptr) {
	m_ptr->flex_rigid.num_nodes = 0;
	m_ptr->other_pairs.num_pairs = 0;

	// P1 is ligand-only co-docking; flexible receptor + multi-ligand is deferred.
	if (m.get_flex().size() > 0)
		throw std::runtime_error("Multi-ligand co-docking with a flexible receptor is not supported yet (P1 is ligand-only).");

	const int natoms = (int)m.atoms.size();
	if (natoms > MAX_NUM_OF_MULTI_ATOMS)
		throw std::runtime_error("Multi-ligand job too large: " + std::to_string(natoms) +
			" movable atoms exceeds MAX_NUM_OF_MULTI_ATOMS=" + std::to_string(MAX_NUM_OF_MULTI_ATOMS));

	for (int ai = 0; ai < natoms; ai++) {
		m_ptr->atoms[ai].types[0] = m.atoms[ai].el;
		m_ptr->atoms[ai].types[1] = m.atoms[ai].ad;
		m_ptr->atoms[ai].types[2] = m.atoms[ai].xs;
		m_ptr->atoms[ai].types[3] = m.atoms[ai].sy;
		for (int j = 0; j < 3; j++) m_ptr->atoms[ai].coords[j] = m.atoms[ai].coords[j];
		m_ptr->atoms[ai].charge = (float)m.atoms[ai].charge;
	}
	for (int ci = 0; ci < (int)m.coords.size(); ci++)
		for (int j = 0; j < 3; j++) m_ptr->m_coords.coords[ci][j] = m.coords[ci].data[j];
	for (int ci = 0; ci < (int)m.minus_forces.size(); ci++)
		for (int j = 0; j < 3; j++) m_ptr->minus_forces.coords[ci][j] = m.minus_forces[ci].data[j];

	const int N = (int)m.ligands.size();
	if (N > MAX_NUM_OF_LIGANDS)
		throw std::runtime_error("Too many ligands: " + std::to_string(N) +
			" exceeds MAX_NUM_OF_LIGANDS=" + std::to_string(MAX_NUM_OF_LIGANDS));
	m_ptr->num_ligands = N;

	// Depth-first tree builder for one ligand (identical logic to fill_m_cl's store_node).
	struct tmp_struct_local {
		int start_index = 0;
		int parent_index = 0;
		void store_node(tree<segment>& child_ptr, rigid_multi_cl& rigid) {
			start_index++;
			rigid.parent[start_index] = parent_index;
			rigid.atom_range[start_index][0] = child_ptr.node.begin;
			rigid.atom_range[start_index][1] = child_ptr.node.end;
			for (int j = 0; j < 9; j++) rigid.orientation_m[start_index][j] = child_ptr.node.get_orientation_m().data[j];
			rigid.orientation_q[start_index][0] = child_ptr.node.orientation().R_component_1();
			rigid.orientation_q[start_index][1] = child_ptr.node.orientation().R_component_2();
			rigid.orientation_q[start_index][2] = child_ptr.node.orientation().R_component_3();
			rigid.orientation_q[start_index][3] = child_ptr.node.orientation().R_component_4();
			for (int j = 0; j < 3; j++) {
				rigid.origin[start_index][j]          = child_ptr.node.get_origin()[j];
				rigid.axis[start_index][j]            = child_ptr.node.get_axis()[j];
				rigid.relative_axis[start_index][j]   = child_ptr.node.relative_axis[j];
				rigid.relative_origin[start_index][j] = child_ptr.node.relative_origin[j];
			}
			if (child_ptr.children.empty()) return;
			if (start_index >= MAX_NUM_OF_RIGID_MULTI) { throw std::runtime_error("Children map too large!"); }
			int parent_index_tmp = start_index;
			for (int ci = 0; ci < (int)child_ptr.children.size(); ci++) {
				this->parent_index = parent_index_tmp;
				this->store_node(child_ptr.children[ci], rigid);
			}
		}
	};

	int torsion_offset = 0;
	for (int k = 0; k < N; k++) {
		ligand m_ligand = m.ligands[k];
		ligand_multi_cl& L = m_ptr->ligands[k];
		rigid_multi_cl& rigid = L.rigid;

		L.begin = m.ligands[k].begin;
		L.end   = m.ligands[k].end;
		if (m_ligand.end > MAX_NUM_OF_MULTI_ATOMS)
			throw std::runtime_error("Ligand atom index exceeds MAX_NUM_OF_MULTI_ATOMS.");

		// root node (index 0)
		rigid.atom_range[0][0] = m_ligand.node.begin;
		rigid.atom_range[0][1] = m_ligand.node.end;
		for (int j = 0; j < 3; j++) rigid.origin[0][j] = m_ligand.node.get_origin()[j];
		for (int j = 0; j < 9; j++) rigid.orientation_m[0][j] = m_ligand.node.get_orientation_m().data[j];
		rigid.orientation_q[0][0] = m_ligand.node.orientation().R_component_1();
		rigid.orientation_q[0][1] = m_ligand.node.orientation().R_component_2();
		rigid.orientation_q[0][2] = m_ligand.node.orientation().R_component_3();
		rigid.orientation_q[0][3] = m_ligand.node.orientation().R_component_4();
		for (int j = 0; j < 3; j++) { rigid.axis[0][j] = 0; rigid.relative_axis[0][j] = 0; rigid.relative_origin[0][j] = 0; }

		tmp_struct_local ts;
		for (int ci = 0; ci < (int)m_ligand.children.size(); ci++) {
			ts.parent_index = 0;
			ts.store_node(m_ligand.children[ci], rigid);
		}
		rigid.num_children = ts.start_index;
		rigid.parent[0] = -1;

		// this ligand owns lig_torsion[torsion_offset .. torsion_offset + num_torsions)
		L.num_torsions   = ts.start_index;
		L.torsion_offset = torsion_offset;
		torsion_offset  += ts.start_index;

		// intra-ligand interacting pairs
		const int np = (int)m.ligands[k].pairs.size();
		if (np > MAX_NUM_OF_LIG_PAIRS_MULTI)
			throw std::runtime_error("Ligand has " + std::to_string(np) +
				" intra pairs, exceeds MAX_NUM_OF_LIG_PAIRS_MULTI=" + std::to_string(MAX_NUM_OF_LIG_PAIRS_MULTI));
		m_ptr->lig_pairs[k].num_pairs = np;
		for (int pi = 0; pi < np; pi++) {
			m_ptr->lig_pairs[k].type_pair_index[pi] = m.ligands[k].pairs[pi].type_pair_index;
			m_ptr->lig_pairs[k].a[pi]               = m.ligands[k].pairs[pi].a;
			m_ptr->lig_pairs[k].b[pi]               = m.ligands[k].pairs[pi].b;
		}
	}
	if (torsion_offset > MAX_NUM_OF_LIG_TORSION)
		throw std::runtime_error("Total ligand torsions " + std::to_string(torsion_offset) +
			" exceed MAX_NUM_OF_LIG_TORSION=" + std::to_string(MAX_NUM_OF_LIG_TORSION));

	// inter-ligand (+ ligand-flex/flex-flex) pairs — already built by model::append()
	{
		const interacting_pairs& op = m.get_other_pairs();
		if ((int)op.size() > MAX_NUM_OF_OTHER_PAIRS)
			throw std::runtime_error("Inter-ligand pair count " + std::to_string(op.size()) +
				" exceeds MAX_NUM_OF_OTHER_PAIRS=" + std::to_string(MAX_NUM_OF_OTHER_PAIRS) +
				" (flat list is a P1 limit; P3 replaces it with a neighbor grid).");
		m_ptr->other_pairs.num_pairs = (int)op.size();
		for (int pi = 0; pi < (int)op.size(); pi++) {
			m_ptr->other_pairs.type_pair_index[pi] = (int)op[pi].type_pair_index;
			m_ptr->other_pairs.a[pi] = (int)op[pi].a;
			m_ptr->other_pairs.b[pi] = (int)op[pi].b;
		}
	}

	m_ptr->m_num_movable_atoms = m.num_movable_atoms();
}
