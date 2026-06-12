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

#ifndef VINA_MODEL_H
#define VINA_MODEL_H

#include <boost/optional.hpp> // for context

#include "file.h"
#include "tree.h"
#include "matrix.h"
#include "precalculate.h"
#include "igrid.h"
#include "grid_dim.h"


/*
*����һ���ṹ������interacting_pair�����ݳ�Ա����unsigned int���͵�type_pair_index��a��b
*/

struct interacting_pair {
	sz type_pair_index;
	sz a;
	sz b;
	interacting_pair(sz type_pair_index_, sz a_, sz b_) : type_pair_index(type_pair_index_), a(a_), b(b_) {}
	//���캯�����������ṹ���Ӧ����ʱ�������б���������unsigned int������ʱ����˳��ֱ��ʼ��type_pair_index��a��b
};

typedef std::vector<interacting_pair> interacting_pairs;
//��interacting_pair��������ȡ����interacting_pairs


//����һ���ṹ������Ϊparsed_line�������һ����Ԫ�ڴ洢����������󣬷ֱ���string��
//��unsigned int���͵�optional�ࣨ������һ�����ܴ��һ��Ԫ�ص���������ʵ����"δ��ʼ��"�ĸ��
//���Ԫ��δ��ʼ������ô�������ǿյģ����������ھ�����Ч�ģ��Ѿ���ʼ����ֵ����

typedef std::pair<std::string, boost::optional<sz> > parsed_line;


typedef std::vector<parsed_line> context;
//��parsed_line��������ȡ����context

/*ligand��,�̳�flexible_body, atom_range
*��Ա����unsigned����degrees_of_freedom��interacting_pairs����pairs��context����cont
*���캯��residue����һ��flexible_body��������f��ʼ��flexible_body����һ��unsigned��degrees_of_freedom��������atom_range�������0��0
*����һ������set_range()
*/
struct ligand : public flexible_body, atom_range {
	unsigned degrees_of_freedom; // can be different from the apparent number of rotatable bonds, because of the disabled torsions
	interacting_pairs pairs;
	context cont;
	ligand(const flexible_body& f, unsigned degrees_of_freedom_) : flexible_body(f), atom_range(0, 0), degrees_of_freedom(degrees_of_freedom_) {}
	void set_range();
};



/*residue��,�̳�main_branch
*���캯��residue����һ��main_branch��������m��ʼ��main_branch
*/
struct residue : public main_branch {
	residue(const main_branch& m) : main_branch(m) {}
};

enum distance_type {DISTANCE_FIXED, DISTANCE_ROTOR, DISTANCE_VARIABLE};
//����ö������distance_type��ö�ٳ�Ա����DISTANCE_FIXED, DISTANCE_ROTOR, DISTANCE_VARIABLE��������ֵ����

//��distance_typeʵ����strictly_triangular_matrix��ģ�壬��ȡ����Ϊdistance_type_matrix
typedef strictly_triangular_matrix<distance_type> distance_type_matrix;

struct non_cache;                 // ��ǰ�����ṹ��non_cache
struct naive_non_cache;           // ��ǰ�����ṹ��naive_non_cache
struct cache;                     // ��ǰ�����ṹ��cache
struct szv_grid;                  // ��ǰ�����ṹ��szv_grid
struct terms;                     // ��ǰ�����ṹ��terms
struct conf_independent_inputs;   // ��ǰ�����ṹ��conf_independent_inputs
struct pdbqt_initializer;         // ��ǰ�����ṹ��pdbqt_initializer - only declared in parse_pdbqt.cpp
struct model_test;                // ��ǰ�����ṹ��model_test



//����ṹ��model

struct model {
    //���г�Ա
	//����append����������model���������
	void append(const model& m);

	//���庯��atom_typing_used������һ��t���͵�����m_atom_typing_used
	atom_type::t atom_typing_used() const { return m_atom_typing_used; }

	sz num_movable_atoms() const { return m_num_movable_atoms; } 
	//����һ������unsigned int�������ݵĺ���num_movable_atoms������˽�г�Աm_num_movable_atoms
	sz num_internal_pairs() const; 
	//����һ������num_other_pairs������other_pairs�����Ĵ�С��unsigned int�������ݣ�
	sz num_other_pairs() const { return other_pairs.size(); }
	//����һ������num_other_pairs������ligands�����Ĵ�С��unsigned int�������ݣ�
	sz num_ligands() const { return ligands.size(); }
	//����һ������num_flex������flex�����Ĵ�С��unsigned int�������ݣ�
	sz num_flex() const { return flex.size(); }
	// GPU flex docking (Stage 1): accessor to populate the GPU flex forest from the model.
	vector_mutable<residue>& get_flex() { return flex; }
	const vector_mutable<residue>& get_flex() const { return flex; }
	// GPU flex docking (Stage 2): accessor for ligand↔flex / flex↔flex interaction pairs.
	const interacting_pairs& get_other_pairs() const { return other_pairs; }

	//����һ������ligand_degrees_of_freedom������һ��unsigned int��������ligand_number������ligands������ligand_number��Ԫ�صĳ�Աdegrees_of_freedom��unsigned int���ͣ�
	sz ligand_degrees_of_freedom(sz ligand_number) const { return ligands[ligand_number].degrees_of_freedom; }
	//����һ������ligand_longest_branch������һ��unsigned int��������ligand_number�����һ��unsigned int���ͷ���ֵ
	sz ligand_longest_branch(sz ligand_number) const;
	//����һ������ligand_length������һ��unsigned int��������ligand_number�����һ��unsigned int���ͷ���ֵ
	sz ligand_length(sz ligand_number) const;

	//����һ����������get_movable_atom_types������һ��t��������atom_typing_used_������һ��unsigned int����
	szv get_movable_atom_types(atom_type::t atom_typing_used_) const;

	//����һ����������get_size������һ��conf_size��
	conf_size get_size() const;
	//����һ����������get_initial_conf������һ��conf��
	conf get_initial_conf() const; // torsions = 0, orientations = identity, ligand positions = current

	//����һ����������movable_atoms_box������double����add_to_each_dimension��granularity������granularityĬ��Ϊ0.375������һ��grid_dims��
	grid_dims movable_atoms_box(fl add_to_each_dimension, fl granularity = 0.375) const;


	/*
	* ������write_flex����������
	* input��path�����������name��string���ͳ�������remark
	* output: ��
	* function������write_context����������model��Աflex_context��name��remark
	*/
	void write_flex  (                  const path& name, const std::string& remark) const { write_context(flex_context, name, remark); }
	
	/*
	* ������write_ligand����������
	* input��unsigned int����ligand_number��path�����������name��string���ͳ�������remark
	* output: ��
	* function�����ligand_number�Ƿ�С��model��Աligands�Ĵ�С���������write_context����������model��Աligands��ligand_number��Ԫ�س�Աcont��name��remark�����򱨴�����ֹ�����ִ��
	*/
	void write_ligand(sz ligand_number, const path& name, const std::string& remark) const { VINA_CHECK(ligand_number < ligands.size()); write_context(ligands[ligand_number].cont, name, remark); }
	
	
	
	/*
	* ������write_structure����������
	* input���ļ������ofile��out
	* output: ��
	* function��1.����ligands����Ԫ�أ��ֱ����write_context����������ligands������i��Ԫ�صĳ�Աcont������out
	* 2.�ж�flex������С�Ƿ����0���������write_context����������flex_context������out
	*/
	void write_structure(ofile& out) const {
		VINA_FOR_IN(i, ligands)
			write_context(ligands[i].cont, out);
		if(num_flex() > 0) // otherwise remark is written in vain
			write_context(flex_context, out);
	}

	/*
	* ����������write_structure����������
	* input���ļ������ofile��out��string��remark
	* output: ��
	* function��1.���ļ���������remark
	* 2.����ԭwrite_structure�����������ļ������out
	*/
	void write_structure(ofile& out, const std::string& remark) const {
		out << remark;
		write_structure(out);
	}

	/*
	* ����������write_structure����������
	* input��path���������name
	* output: ��
	* function��1.ͨ��path�ഴ���ļ������out
	* 2.����ԭwrite_structure�����������ļ������out
	*/
	void write_structure(const path& name) const { ofile out(name); write_structure(out); }

	/*
	* ������write_model����������
	* input���ļ������ofile��out��unsigned int��model_number��string��remark
	* output: ��
	* function�����ļ����������ַ�����MODEL model_number�����У�
	* 2.����write_structure���ذ汾�����������ļ������out��remark
	* 3.���ļ����������ַ�����ENDMDL�����У�
	*/
	void write_model(ofile& out, sz model_number, const std::string& remark) const {
		out << "MODEL " << model_number << '\n';
		write_structure(out, remark);
		out << "ENDMDL\n";
	}

	//��������seti������һ��conf�����ã��޷���ֵ
	void seti(const conf& c);
	//��������sete������һ��conf�����ã��޷���ֵ
	void sete(const conf& c);
	//��������set������һ��conf�����ã��޷���ֵ
	void set (const conf& c);

	//������������ gyration_radius������һ��unsigned int��ligand_number������double��
	fl gyration_radius(sz ligand_number) const; // uses coords

	/*
	* ������movable_atom����������
	* input��unsigned int��i
	* output: atom_base������
	* function���ж�i���Ƿ�С��m_num_movable_atoms�����򷵻�atoms������i��Ԫ�أ����򱨴�����ֹ�����ִ��
	*/
	const atom_base& movable_atom  (sz i) const { assert(i < m_num_movable_atoms); return  atoms[i]; }

	/*
	* ������movable_coords����������
	* input��unsigned int��i
	* output: ����Ϊ3��ʸ��vec
	* function���ж�i���Ƿ�С��m_num_movable_atoms�����򷵻�coords������i��Ԫ�أ����򱨴�����ֹ�����ִ��
	*/
	const vec&       movable_coords(sz i) const { assert(i < m_num_movable_atoms); return coords[i]; }


	//������������atom_coords������atom_index�ೣ�����ã�����vecʸ����������
	const vec& atom_coords(const atom_index& i) const;

	//������������distance_sqr_between����������atom_index�ೣ�����ã�����double������
	fl distance_sqr_between(const atom_index& a, const atom_index& b) const;

	//������������atom_exists_between������distance_type_matrix�ೣ�����ã�����atom_index�ೣ�����ã�unsigned int���������������ã�����һ������ֵ
	bool atom_exists_between(const distance_type_matrix& mobility, const atom_index& a, const atom_index& b, const szv& relevant_atoms) const; // there is an atom closer to both a and b then they are to each other and immobile relative to them

	//������������distance_type_between������distance_type_matrix�ೣ�����ã�����atom_index�ೣ�����ã�����һ��ö������distance_type
	distance_type distance_type_between(const distance_type_matrix& mobility, const atom_index& i, const atom_index& j) const;

	// clean up
	//������������evali������precalculate�ೣ�����ã�vecʸ���������ã�����double������
	fl evali     (const precalculate& p,                  const vec& v                          ) const;
	//������������evale������precalculate�ೣ�����ã�igrid�ೣ�����ã�vecʸ���������ã�����double������
	fl evale     (const precalculate& p, const igrid& ig, const vec& v                          ) const;
	//��������eval������precalculate�ೣ�����ã�igrid�ೣ�����ã�vecʸ���������ã�conf�ೣ�����ã�����double������
	fl eval      (const precalculate& p, const igrid& ig, const vec& v, const conf& c           );
	//��������eval_deriv������precalculate�ೣ�����ã�igrid�ೣ�����ã�vecʸ���������ã�conf�ೣ�����ã�change�����ã�����double������
	fl eval_deriv(const precalculate& p, const igrid& ig, const vec& v, const conf& c, change& g);


	//��������eval_intramolecular������precalculate�ೣ�����ã�vecʸ���������ã�conf�ೣ�����ã�����double������
	fl eval_intramolecular(                            const precalculate& p,                  const vec& v, const conf& c);
	//��������eval_adjusted������scoring_function�ೣ�����ã�����precalculate�ೣ�����ã�igrid�ೣ�����ã�vecʸ���������ã�conf�ೣ�����ã�double������������double������
	fl eval_adjusted      (const scoring_function& sf, const precalculate& p, const igrid& ig, const vec& v, const conf& c, fl intramolecular_energy);

	//������������rmsd_lower_bound������model����󣬷���double������
	fl rmsd_lower_bound(const model& m) const; // uses coords
	//������������rmsd_upper_bound������model����󣬷���double������
	fl rmsd_upper_bound(const model& m) const; // uses coords
	//������������rmsd_ligands_upper_bound������model����󣬷���double������
	fl rmsd_ligands_upper_bound(const model& m) const; // uses coords

	//������������verify_bond_lengths���޷���
	void verify_bond_lengths() const;
	//������������about���޷���
	void about() const;


	/*
	* ������get_ligand_internal_coords����������
	* input����
	* output: vec��������
	* function��1.�ж�model��Աligands�����Ĵ�С�Ƿ����1����������ִ�У����򱨴�����ֹ�����ִ��
	* 2.����һ����ʱvecv����tmp
	* 3.����model��Աligands��������ʼԪ�ص�����,����lig
	* 4.i��lig��Աbegin�Ĵ�С������lig��Աbegin�Ĵ�С��������tmpβ������model��Աinternal_coords������i��Ԫ��
	* 5.����tmp
	*/
	vecv get_ligand_internal_coords() const { // FIXME rm
		VINA_CHECK(ligands.size() == 1);
		vecv tmp;
		const ligand& lig = ligands.front();
		VINA_RANGE(i, lig.begin, lig.end)
			tmp.push_back(internal_coords[i]);
		return tmp;
	}


	/*
	* ������get_ligand_coords����������
	* input����
	* output: vec��������
	* function��1.�ж�model��Աligands�����Ĵ�С�Ƿ����1����������ִ�У����򱨴�����ֹ�����ִ��
	* 2.����һ����ʱvecv����tmp
	* 3.����model��Աligands��������ʼԪ�ص�����,����lig
	* 4.i��lig��Աbegin�Ĵ�С������lig��Աbegin�Ĵ�С��������tmpβ������model��Աcoords������i��Ԫ��
	* 5.����tmp
	*/
	vecv get_ligand_coords() const { // FIXME rm
		VINA_CHECK(ligands.size() == 1);
		vecv tmp;
		const ligand& lig = ligands.front();
		VINA_RANGE(i, lig.begin, lig.end)
			tmp.push_back(coords[i]);
		return tmp;
	}

	/*
	* ������get_heavy_atom_movable_coords����������
	* input����
	* output: vec��������
	* function��1.����һ����ʱvecv����tmp
	* 2.i��0������m_num_movable_atoms�������ж�atoms��i��Ԫ�صĳ�Աel�Ƿ����0��������tmpβ������model��Աcoords������i��Ԫ��
	* 3.����tmp
	*/
	vecv get_heavy_atom_movable_coords() const { // FIXME mv
		vecv tmp;
		VINA_FOR(i, num_movable_atoms())
			if(atoms[i].el != EL_TYPE_H)
				tmp.push_back(coords[i]);
		return tmp;
	}

	//����һ����������check_internal_pairs���޷���
	void check_internal_pairs() const;
	//����һ����������print_stuff���޷���
	void print_stuff() const; // FIXME rm

	//����һ����������clash_penalty������double��������
	fl clash_penalty() const;

	//˽�г�Ա
private:
	friend struct non_cache;              // ������Ԫ�ṹ��non_cache��                        �˽ṹ��ɷ��ʽṹ��model()��˽�г�Ա                                           
	friend struct naive_non_cache;		  // ������Ԫ�ṹ��naive_non_cache��                  �˽ṹ��ɷ��ʽṹ��model()��˽�г�Ա 			
	friend struct cache;				  // ������Ԫ�ṹ��cache��                            �˽ṹ��ɷ��ʽṹ��model()��˽�г�Ա 					
	friend struct szv_grid;				  // ������Ԫ�ṹ��szv_grid	��                        �˽ṹ��ɷ��ʽṹ��model()��˽�г�Ա 			
	friend struct terms;				  // ������Ԫ�ṹ��terms��                            �˽ṹ��ɷ��ʽṹ��model()��˽�г�Ա 					
	friend struct conf_independent_inputs;// ������Ԫ�ṹ��conf_independent_inputs��          �˽ṹ��ɷ��ʽṹ��model()��˽�г�Ա 	
	friend struct appender_info;		  // ������Ԫ�ṹ��appender_info��                    �˽ṹ��ɷ��ʽṹ��model()��˽�г�Ա 			
	friend struct pdbqt_initializer;	  // ������Ԫ�ṹ��pdbqt_initializer��                �˽ṹ��ɷ��ʽṹ��model()��˽�г�Ա  - only declared in parse_pdbqt.cpp
	friend struct model_test;			  // ������Ԫ�ṹ��model_test��                       �˽ṹ��ɷ��ʽṹ��model()��˽�г�Ա 

	//���캯��model()����ʼ��m_num_movable_atomsΪ0��m_atom_typing_usedΪXS
	model() : m_num_movable_atoms(0), m_atom_typing_used(atom_type::XS) {};


	/*
	* ������get_atom���ǳ�������
	* input��atom_index������i
	* function���ж�i�ĳ�Աin_grid�Ƿ�Ϊ�棬���򷵻�grid_atoms�����ĵڣ�����i�ĳ�Աi����ֵ��Ԫ�س������ã����򷵻�atoms�����ĵڣ�����i�ĳ�Աi����ֵ��Ԫ�س�������
	*/
	const atom& get_atom(const atom_index& i) const { return (i.in_grid ? grid_atoms[i.i] : atoms[i.i]); }
	/*
	* ������get_atom
	* input��atom_index������i
	* function���ж�i�ĳ�Աin_grid�Ƿ�Ϊ�棬���򷵻�grid_atoms�����ĵڣ�����i�ĳ�Աi����ֵ��Ԫ�����ã����򷵻�atoms�����ĵڣ�����i�ĳ�Աi����ֵ��Ԫ������
	*/
	      atom& get_atom(const atom_index& i)       { return (i.in_grid ? grid_atoms[i.i] : atoms[i.i]); }


	//1.����һ����������write_context������context���������һ���ļ���������޷���
	void write_context(const context& c, ofile& out) const;
	


	/*2.���س�������write_context������context��������һ���ļ��������string��remark���޷���
	* function:���ļ���������remark
	*/
	void write_context(const context& c, ofile& out, const std::string& remark) const {
		out << remark;
	}

	/*3.���س�������write_context������context���������һ���ļ�path������޷���
	* function:ͨ��path�ഴ��һ���ļ������out,�ٵ���write_context������1���汾
	*/
	void write_context(const context& c, const path& name) const {
		ofile out(name);
		write_context(c, out);
	}

	/*4.���س�������write_context������context��������һ���ļ�path������string��remark���޷���
	* function:ͨ��path�ഴ��һ���ļ������out,�ٵ���write_context������2���汾
	*/
	void write_context(const context& c, const path& name, const std::string& remark) const {
		ofile out(name);
		write_context(c, out, remark);
	}

	//����һ����������rmsd_lower_bound_asymmetric������model��x,y���ã�����һ��double������
	fl rmsd_lower_bound_asymmetric(const model& x, const model& y) const; // actually static
	

	//����һ����������sz_to_atom_index������unsigned int��������i������һ��atom_index��
	atom_index sz_to_atom_index(sz i) const; // grid_atoms, atoms
	//����һ����������bonded_to_HD������atom������a�����ز���ֵ
	bool bonded_to_HD(const atom& a) const;
	//����һ����������bonded_to_heteroatom������atom������a�����ز���ֵ
	bool bonded_to_heteroatom(const atom& a) const;
	//����һ����������find_ligand������unsigned int��������a������һ��unsigned int��������
	sz find_ligand(sz a) const;
	//����һ����������bonded_to������unsigned int��������a��n����unsigned int������������szv���޷���
	void bonded_to(sz a, sz n, szv& out) const;
	//���س�������bonded_to������unsigned int��������a��n������unsigned int��������
	szv bonded_to(sz a, sz n) const;

	//����һ������assign_bonds������distance_type_matrix�����ã��޷���
	void assign_bonds(const distance_type_matrix& mobility); // assign bonds based on relative mobility, distance and covalent length
	//����һ������assign_types���޷���
	void assign_types();
	//����һ������initialize_pairs������distance_type_matrix�����ã��޷���
	void initialize_pairs(const distance_type_matrix& mobility);
	//����һ������initialize������distance_type_matrix�����ã��޷���
	void initialize(const distance_type_matrix& mobility);
	//����һ����������clash_penalty_aux������interacting_pair���������ã�����double����
	fl clash_penalty_aux(const interacting_pairs& pairs) const;

	vecv internal_coords; //����һ������internal_coords����������������ɽṹ��vec������Ϊ3��ʸ����
	




	


	vector_mutable<residue> flex; //����һ��vector_mutable����flex����Ԫ��Ϊresidue��
	context flex_context;//����һ��parsed_line������������

	//����һ��interacting_pairs����other_pairs
	interacting_pairs other_pairs; // all except internal to one ligand: ligand-other ligands; ligand-flex/inflex; flex-flex/inflex

	sz m_num_movable_atoms; //����һ��unsigned int��������
	atom_type::t m_atom_typing_used;//����һ��t�������ݣ�t��ö������
public:
		vector_mutable<ligand> ligands; //����һ��vector_mutable����ligands����Ԫ��Ϊligand��
		atomv atoms; // movable, inflex
		vecv coords;          //����һ������coords��         ��������������ɽṹ��vec������Ϊ3��ʸ����
		vecv minus_forces;    //����һ������minus_forces��   ��������������ɽṹ��vec������Ϊ3��ʸ����	
		atomv grid_atoms;//��������grid_atoms��atoms��Ԫ�ؾ�Ϊatom��
};


#endif

