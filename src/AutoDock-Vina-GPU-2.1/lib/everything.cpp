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

#include "everything.h"
#include "int_pow.h"

// Definition of active_xs_radii — defaults to Vina radii; main() may point to Vinardo radii.
const fl* active_xs_radii = xs_vdw_radii;

// Global scoring type — set once before precalculation.
ScoringType g_scoring_type = ScoringType::Vina;



// distance_additive terms   ����double���͵�x   width  ���e�ģ�-��x/width����ƽ���� double
inline fl gaussian(fl x, fl width) {
	return std::exp(-sqr(x/width));
}


/*�ṹ��electrostatic�̳�distance_additive    ����ģ�����UINT����i
* ����double   cap
* ����electrostatic�������е�string����name��ֵ
* ����double���ͺ���eval  ����ṹ��a��b  double r  ���double      
	����charge1*charge2 * cap����r��i�η���֮һ��cap�Ľ�Сֵ
*/
template<unsigned i>
struct electrostatic : public distance_additive {
	fl cap;
	electrostatic(fl cap_, fl cutoff_) : distance_additive(cutoff_), cap(cap_) {
		name = std::string("electrostatic(i=") + to_string(i) + ", ^=" + to_string(cap) + ", c=" + to_string(cutoff) + ")";
	}
	fl eval(const atom_base& a, const atom_base& b, fl r) const {
		fl tmp = int_pow<i>(r);   //r��i�η�
		fl q1q2 = a.charge * b.charge;//charge1*charge2
		if(tmp < epsilon_fl) return q1q2 * cap;//epsilon_fl   2.2204460492503131e-016   
		else                 return q1q2 * (std::min)(cap, 1/int_pow<i>(r));//����  r��i�η���֮һ��cap�Ľ�Сֵ
	}
};        
/*
function��
���룺�ṹ��  ԭ������  ���  double��������
* �ж�  ����ṹ��Ĳ���ad<16 �򷵻�����atom_kind_data[ad]�е�solvation  �����ǽṹ�����͵�
 ����ṹ��Ĳ���xs<16�򷵻�-0.00110

*/
fl solvation_parameter(const atom_type& a) {
	if(a.ad < AD_TYPE_SIZE) return ad_type_property(a.ad).solvation;//ad<20    ��������atom_kind_data[ad]�е�solvation  �����ǽṹ�����͵�
	else if(a.xs == XS_TYPE_Met_D) return metal_solvation_parameter;//metal_solvation_parameter = -0.00110
	VINA_CHECK(false); 
	return 0; // placating the compiler
}                
/*
function��
���룺�ṹ��  ԭ������  ���  double��������
* �ж�  ����ṹ��Ĳ���ad<16 �򷵻�����atom_kind_data[ad]�е�volume  �����ǽṹ�����͵�
 ����ṹ��Ĳ���xs<17�򷵻� ��xs_vdw_radii[xs]�������η�*4*pi / 3  

*/
fl volume(const atom_type& a) {
	if(a.ad < AD_TYPE_SIZE) return ad_type_property(a.ad).volume;
	else if(a.xs < XS_TYPE_SIZE) return 4*pi / 3 * int_pow<3>(xs_radius(a.xs));
	VINA_CHECK(false);
	return 0; // placating the compiler
}                

/*�ṹ��ad4_solvation �̳�distance_additive    
* ����double    desolvation_sigma;solvation_q  bool����  charge_dependent
* ����ad4_solvation�������е�string����name��ֵ
* ����double���ͺ���eval  ����ṹ��a��b  double r  
	����double���͵���tmp
*/
struct ad4_solvation : public distance_additive {
	fl desolvation_sigma;
	fl solvation_q;
	bool charge_dependent;
	ad4_solvation(fl desolvation_sigma_, fl solvation_q_, bool charge_dependent_, fl cutoff_) : distance_additive(cutoff_), solvation_q(solvation_q_), charge_dependent(charge_dependent_), desolvation_sigma(desolvation_sigma_) {
		name = std::string("ad4_solvation(d-sigma=") + to_string(desolvation_sigma) + ", s/q=" + to_string(solvation_q) + ", q=" + to_string(charge_dependent) + ", c=" + to_string(cutoff) + ")";
	}//   name����ֵ ad4_solvation(d-sigma=    ,s/q=   ,q=     ,c=    )��
	fl eval(const atom_base& a, const atom_base& b, fl r) const {
		fl q1 = a.charge;
		fl q2 = b.charge;

		VINA_CHECK(not_max(q1));//q1<0.17976931348623158e+308
		VINA_CHECK(not_max(q2));//q2<0.17976931348623158e+308

		sz t1 = a.ad;
		sz t2 = b.ad;

		fl solv1 = solvation_parameter(a);
		fl solv2 = solvation_parameter(b);

		fl volume1 = volume(a);
		fl volume2 = volume(b);

		fl my_solv = charge_dependent ? solvation_q : 0;

		fl tmp = ((solv1 + my_solv * std::abs(q1)) * volume2 + 
			    (solv2 + my_solv * std::abs(q2)) * volume1) * std::exp(-sqr(r/(2*desolvation_sigma)));

		VINA_CHECK(not_max(tmp));//tmp<0.17976931348623158e+308
		return tmp;
	}
};
/*
����double���͵ĺ���    ����  uint xs_t1��xs_t2   ����double �������е����� xs_vdw_radii[xs_t1]+xs_vdw_radii[xs_t2]
*/
inline fl optimal_distance(sz xs_t1, sz xs_t2) {
	return xs_radius(xs_t1) + xs_radius(xs_t2);    
}
/*�ṹ��gauss�̳�usable    
* ����double   offset   width
* ����gauss�������е�string����name��ֵ   name����ֵ gauss(o=     ,w=    ,c=     )
* ����double���ͺ���eval  ����double����t1   t2  r     ���double
	����e��-{(r-xs_vdw_radii[xs_t1]-xs_vdw_radii[xs_t2]-offset)/width}��ƽ��
*/
struct gauss : public usable {
	fl offset; // added to optimal distance
	fl width;
	gauss(fl offset_, fl width_, fl cutoff_) : usable(cutoff_), offset(offset_), width(width_) {
		name = std::string("gauss(o=") + to_string(offset) + ", w=" + to_string(width) + ", c=" + to_string(cutoff) + ")";
	}//name����ֵ gauss(o=, w=, c = )
	fl eval(sz t1, sz t2, fl r) const {
		return gaussian(r - (optimal_distance(t1, t2) + offset), width);
	}// e��-{(r-xs_vdw_radii[xs_t1]-xs_vdw_radii[xs_t2]-offset)/width}��ƽ��
};

/*�ṹ��repulsion�̳�usable
* ����double   offset   
* ����gauss�������е�string����name��ֵ   name����ֵ repulsion(o=     )
* ����double���ͺ���eval  ����double����t1   t2  r     ���double
	���أ�r-xs_vdw_radii[t1]-xs_vdw_radii[t2]-offset����ƽ��
*/


struct repulsion : public usable {
	fl offset; // added to vdw
	repulsion(fl offset_, fl cutoff_) : usable(cutoff_), offset(offset_) {
		name = std::string("repulsion(o=") + to_string(offset) + ")";
	}
	fl eval(sz t1, sz t2, fl r) const {
		fl d = r - (optimal_distance(t1, t2) + offset);//d=r-xs_vdw_radii[t1]-xs_vdw_radii[t2]-offset
		if(d > 0) //�ж�d>0
			return 0;
		return d*d;//����d��ƽ��
	}
};
//����double   a��b��c    �жϣ�c<a<b����c<=b<=a����0      a<b<c����b<=a<=c����1   ������ ����double��c-a��/(b-a)
inline fl slope_step(fl x_bad, fl x_good, fl x) {   
	if(x_bad < x_good) {
		if(x <= x_bad) return 0;
		if(x >= x_good) return 1;
	}
	else {
		if(x >= x_bad) return 0;
		if(x <= x_good) return 1;
	}
	return (x - x_bad) / (x_good - x_bad);
}


/*�ṹ��hydrophobic�̳�usable
* ����double   good  bad
* ����hydrophobic�������е�string����name��ֵ   name����ֵ hydrophobic(g=     ,b=     ,c=     )
* ����double���ͺ���eval  ����double����t1   t2  r     ���double  �ж� t1=0��12��13��14��15  ����  t2=0��12��13��14��15 �ж�slope_step����������
	����  (��r-xs_vdw_radii[t1]-xs_vdw_radii[t2])-bad��/(good-bad)
*/
struct hydrophobic : public usable {
	fl good;
	fl bad;
	hydrophobic(fl good_, fl bad_, fl cutoff_) : usable(cutoff_), good(good_), bad(bad_) {
		name = "hydrophobic(g=" + to_string(good) + ", b=" + to_string(bad) + ", c=" + to_string(cutoff) + ")";
	}//name����ֵ hydrophobic(g=     ,b=     ,c=     )
	fl eval(sz t1, sz t2, fl r) const {
		if(xs_is_hydrophobic(t1) && xs_is_hydrophobic(t2))//�ж� t1=0��12��13��14��15  ����  t2=0��12��13��14��15
			return slope_step(bad, good, r - optimal_distance(t1, t2));//   (��r-xs_vdw_radii[t1]-xs_vdw_radii[t2])-bad��/(good-bad)
		else return 0;
	}
};


/*�ṹ��non_hydrophobic�̳�usable
* ����double   good  bad
* ����hydrophobic�������е�string����name��ֵ   name����ֵ non_hydrophobic(g=     ,b=     ,c=     )
* ����double���ͺ���eval  ����double����t1   t2  r     ���double  �ж� t1!=0��12��13��14��15  ����  t2!=0��12��13��14��15 �ж�slope_step���������� 
	����  (��r-xs_vdw_radii[t1]-xs_vdw_radii[t2])-bad��/(good-bad)
*/
struct non_hydrophobic : public usable {
	fl good;
	fl bad;
	non_hydrophobic(fl good_, fl bad_, fl cutoff_) : usable(cutoff_), good(good_), bad(bad_) {
		name = "non_hydrophobic(g=" + to_string(good) + ", b=" + to_string(bad) + ", c=" + to_string(cutoff) + ")";
	}
	fl eval(sz t1, sz t2, fl r) const {
		if(!xs_is_hydrophobic(t1) && !xs_is_hydrophobic(t2))
			return slope_step(bad, good, r - optimal_distance(t1, t2));
		else return 0;
	}
};
/*����ģ����� m n uint   ����  double    position  depth c_n   c_m  ���  �ı����c_n   c_m
//c_n = position��n�η�  * depth * m / (fl(n)-fl(m));
//c_m = position��m�η�  * depth * n/ (fl(m)-fl(n));
*/
template<unsigned n, unsigned m>
void find_vdw_coefficients(fl position, fl depth, fl& c_n, fl& c_m) {
	BOOST_STATIC_ASSERT(n != m); //ȷ��n������m
	c_n = int_pow<n>(position) * depth * m / (fl(n)-fl(m));
	c_m = int_pow<m>(position) * depth * n / (fl(m)-fl(n));
}

/*����ģ����� i  j uint 
�ṹ��vdw�̳�usable
* ����double   smoothing  cap
* ����vdw�������е�string����name��ֵ   name����ֵ vdw(i=     ,j=     ,s=     ,^=     ,c=      )
* ����double���ͺ���eval  ����double����t1   t2  r    ����inter  double d0  depth  c_i  c_j  
���double  min(cap, c_i / r��i�η� + c_j / r��j�η�)����cap
*/
template<unsigned i, unsigned j>
struct vdw : public usable {
	fl smoothing;
	fl cap;
	vdw(fl smoothing_, fl cap_, fl cutoff_) 
		: usable(cutoff_), smoothing(smoothing_), cap(cap_) {
		name = "vdw(i=" + to_string(i) + ", j=" + to_string(j) + ", s=" + to_string(smoothing) + ", ^=" + to_string(cap) + ", c=" + to_string(cutoff) + ")";
	}
	fl eval(sz t1, sz t2, fl r) const {
		fl d0 = optimal_distance(t1, t2);//xs_vdw_radii[t1]+xs_vdw_radii[t2]
		fl depth = 1; 
		fl c_i = 0;
		fl c_j = 0;
		find_vdw_coefficients<i, j>(d0, depth, c_i, c_j);//�ı�c_i  c_j
		if     (r > d0 + smoothing) r -= smoothing;//�ı�r
		else if(r < d0 - smoothing) r += smoothing;
		else r = d0;

		fl r_i = int_pow<i>(r);//r��i�η�
		fl r_j = int_pow<j>(r);//r��j�η�
		if(r_i > epsilon_fl && r_j > epsilon_fl)//r��i�η���j�η�ͬʱ����2.2204460492503131e-016
			return (std::min)(cap, c_i / r_i + c_j / r_j);//���ؽ�Сֵ
		else 
			return cap;
	}
};



/*�ṹ��non_dir_h_bond�̳�usable
* ����double   good  bad
* ����non_dir_h_bond�������е�string����name��ֵ   name����ֵ non_dir_h_bond(g=     ,b=    )
* ����double���ͺ���eval  ����double����t1   t2  r     ���double  ��(t1=3��5��7��9��16  ����t2=4��5��8��9)  ����  (t2=3��5��7��9��16  ����t1=4��5��8��9) 
�ж�slope_step����������   ����  (��r-xs_vdw_radii[t1]-xs_vdw_radii[t2])-bad��/(good-bad)
*/
struct non_dir_h_bond : public usable {
	fl good;
	fl bad;
	non_dir_h_bond(fl good_, fl bad_, fl cutoff_) : usable(cutoff_), good(good_), bad(bad_) {
		name = std::string("non_dir_h_bond(g=") + to_string(good) + ", b=" + to_string(bad) + ")";
	}
	fl eval(sz t1, sz t2, fl r) const {
		if(xs_h_bond_possible(t1, t2))//t1=3��5��7��9��16  ����t2=4��5��8��9  ����  t2=3��5��7��9��16  ����t1=4��5��8��9
			return slope_step(bad, good, r - optimal_distance(t1, t2));  //(��r-xs_vdw_radii[t1]-xs_vdw_radii[t2])-bad��/(good-bad)
		return 0;
	}
};
/*���� ����read_iterator   ����   i  ������     ���double
* function   ���ص�����ָ���������Ԫ��  ������ָ����һ��Ԫ��
*/
inline fl read_iterator(flv::const_iterator& i) {
	fl x = *i; 
	++i;
	return x;
}
/*���� ����smooth_div   ����   x   y   double      ���double
* function   x�ľ���ֵ<2.2204460492503131e-016  ����0   y�ľ���ֵ<2.2204460492503131e-016   ����xyͬ�ŷ��� 1.7976931348623158e+308         ��ŷ���- 1.7976931348623158e+308
���򷵻�  x/y
*/
fl smooth_div(fl x, fl y) {
	if(std::abs(x) < epsilon_fl) return 0;
	if(std::abs(y) < epsilon_fl) return ((x*y > 0) ? max_fl : -max_fl); // FIXME I hope -max_fl does not become NaN
	return x / y;
}//x�ľ���ֵ<2.2204460492503131e-016  ����0   y�ľ���ֵ<2.2204460492503131e-016   ����xyͬ�ŷ��� 1.7976931348623158e+308         ��ŷ���- 1.7976931348623158e+308


/*�ṹ��num_tors_add�̳�conf_independent
* ����non_dir_h_bond�������е�string����name��ֵ   name����ֵ num_tors_add
* ����uint����     ����1  ֻ��
* ����double���ͺ���eval  ����ṹ��  in   double����  x    ������  i 
* w=��������ָ���ֵ��������ָ����һ��Ԫ��
���double  x + w * num_tors(�ṹ��Ĳ���);
*/

struct num_tors_add : public conf_independent {
	num_tors_add() { name = "num_tors_add"; }
	sz size() const { return 1; }
	fl eval(const conf_independent_inputs& in, fl x, flv::const_iterator& i) const {
		//fl w = 0.1 * read_iterator(i); // [-1 .. 1]
		fl w = read_iterator(i); // FIXME?
		return x + w * in.num_tors;
	}
};


/*�ṹ��num_tors_sqr�̳�conf_independent
* ����num_tors_sqr�������е�string����name��ֵ   name����ֵ num_tors_sqr
* ����uint����     ����1  ֻ��
* ����double���ͺ���eval  ����ṹ��  in   double����  x    ������  i
* * w=0.1*��������ָ���ֵ��������ָ����һ��Ԫ��
���double  x + w * ��num_tors��ƽ��) / 5;
*/
struct num_tors_sqr : public conf_independent {
	num_tors_sqr() { name = "num_tors_sqr"; }
	sz size() const { return 1; }
	fl eval(const conf_independent_inputs& in, fl x, flv::const_iterator& i) const {
		fl w = 0.1 * read_iterator(i); // [-1 .. 1]
		return x + w * sqr(fl(in.num_tors)) / 5;
	}
};

/*�ṹ��num_tors_sqrt�̳�conf_independent
* ����num_tors_sqrt�������е�string����name��ֵ   name����ֵ num_tors_sqrt
* ����uint����     ����1  ֻ��
* ����double���ͺ���eval  ����ṹ��  in   double����  x    ������  i
* w=0.1*��������ָ���ֵ��������ָ����һ��Ԫ��
���double  x + w * ��num_tors��ƽ����) /��5.0��ƽ����) ;
*/
struct num_tors_sqrt : public conf_independent {
	num_tors_sqrt() { name = "num_tors_sqrt"; }
	sz size() const { return 1; }
	fl eval(const conf_independent_inputs& in, fl x, flv::const_iterator& i) const {
		fl w = 0.1 * read_iterator(i); // [-1 .. 1]
		return x + w * std::sqrt(fl(in.num_tors)) / sqrt(5.0);
	}
};
/*�ṹ��num_tors_div�̳�conf_independent
* ����num_tors_div�������е�string����name��ֵ   name����ֵ num_tors_div
* ����uint����     ����1  ֻ��
* ����double���ͺ���eval  ����ṹ��  in   double����  x    ������  i
w=0.1*��������ָ���ֵ+1�� ������ָ����һ��Ԫ��
��x�ľ���ֵ<2.2204460492503131e-016  ����0   ��(1+w*num_tors/5.0)�ľ���ֵ<2.2204460492503131e-016   ����xyͬ�ŷ��� 1.7976931348623158e+308         ��ŷ���- 1.7976931348623158e+308
���򷵻�  x/ (1+w*num_tors/5.0)
*/
struct num_tors_div : public conf_independent {
	num_tors_div() { name = "num_tors_div"; }
	sz size() const { return 1; }
	fl eval(const conf_independent_inputs& in, fl x, flv::const_iterator& i) const {
		fl w = 0.1 * (read_iterator(i) + 1); // w is in [0..0.2]
		return smooth_div(x, 1 + w * in.num_tors/5.0);
	}
};
/*�ṹ��ligand_length�̳�conf_independent
* ����ligand_length�������е�string����name��ֵ   name����ֵ ligand_length
* ����uint����     ����1  ֻ��
* ����double���ͺ���eval  ����ṹ��  in   double����  x    ������  i
* w=(������ָ���ֵ�� ������ָ����һ��Ԫ��
����double    x + w * in.ligand_lengths_sum
*/
struct ligand_length : public conf_independent {
	ligand_length() { name = "ligand_length"; }
	sz size() const { return 1; }
	fl eval(const conf_independent_inputs& in, fl x, flv::const_iterator& i) const {
		fl w = read_iterator(i);
		return x + w * in.ligand_lengths_sum;
	}
};
/*�ṹ��num_ligands�̳�conf_independent
* ����num_ligands�������е�string����name��ֵ   name����ֵ num_ligands
* ����uint����     ����1  ֻ��
* ����double���ͺ���eval  ����ṹ��  in   double����  x    ������  i
*  w=1*��������ָ���ֵ�� ������ָ����һ��Ԫ��
* ����x + w * in.num_ligands
*/
struct num_ligands : public conf_independent {
	num_ligands() { name = "num_ligands"; }
	sz size() const { return 1; }
	fl eval(const conf_independent_inputs& in, fl x, flv::const_iterator& i) const {
		fl w = 1 * read_iterator(i); // w is in [-1.. 1]
		return x + w * in.num_ligands;
	}
};
/*�ṹ��num_heavy_atoms_div�̳�conf_independent
* ����num_heavy_atoms_div�������е�string����name��ֵ   name����ֵ num_heavy_atoms_div
* ����uint����     ����1  ֻ��
* ����double���ͺ���eval  ����ṹ��  in   double����  x    ������  i
*   w=0.05*��������ָ���ֵ�� ������ָ����һ��Ԫ��
��x�ľ���ֵ<2.2204460492503131e-016  ����0   ��(1+w*num_heavy_atoms)�ľ���ֵ<2.2204460492503131e-016   ����xyͬ�ŷ��� 1.7976931348623158e+308         ��ŷ���- 1.7976931348623158e+308
���򷵻�  x/ (1+w*num_heavy_atoms)
*/
struct num_heavy_atoms_div : public conf_independent {
	num_heavy_atoms_div() { name = "num_heavy_atoms_div"; }
	sz size() const { return 1; }
	fl eval(const conf_independent_inputs& in, fl x, flv::const_iterator& i) const {
		fl w = 0.05 * read_iterator(i); 
		return smooth_div(x, 1 + w * in.num_heavy_atoms); 
	}
};
/*�ṹ��num_heavy_atoms�̳�conf_independent
* ����num_heavy_atoms�������е�string����name��ֵ   name����ֵ num_heavy_atoms
* ����uint����     ����1  ֻ��
* ����double���ͺ���eval  ����ṹ��  in   double����  x    ������  i
* *   w=0.05*��������ָ���ֵ�� ������ָ����һ��Ԫ��
����double   x + w * num_heavy_atoms;
*/
struct num_heavy_atoms : public conf_independent {
	num_heavy_atoms() { name = "num_heavy_atoms"; }
	sz size() const { return 1; }
	fl eval(const conf_independent_inputs& in, fl x, flv::const_iterator& i) const {
		fl w = 0.05 * read_iterator(i); 
		return x + w * in.num_heavy_atoms;
	}
};
/*�ṹ��num_hydrophobic_atoms�̳�conf_independent
* ����num_hydrophobic_atoms�������е�string����name��ֵ   name����ֵ num_hydrophobic_atoms
* ����uint����     ����1  ֻ��
* ����double���ͺ���eval  ����ṹ��  in   double����  x    ������  i
* w=0.05*��������ָ���ֵ�� ������ָ����һ��Ԫ��
* ����double  x + w * in.num_hydrophobic_atoms
*/
struct num_hydrophobic_atoms : public conf_independent {
	num_hydrophobic_atoms() { name = "num_hydrophobic_atoms"; }
	sz size() const { return 1; }
	fl eval(const conf_independent_inputs& in, fl x, flv::const_iterator& i) const {
		fl w = 0.05 * read_iterator(i); 
		return x + w * in.num_hydrophobic_atoms;
	}
};
/*���� �ṹ��everything�� ���캯��everything  
* ���� uint  d    double   cutoff
*/
everything::everything(ScoringType st) {
	const fl cutoff = 8;

	if (st == ScoringType::Vinardo) {
		// Vinardo scoring function (Quiroga & Villarreal, PLOS ONE 2016)
		// 1 Gaussian (wider), no repulsion shift, different hydrophobic ramp, tighter h-bond
		// Atom radii must already be set to xs_vdw_radii_vinardo by caller.
		// Weights: -0.045, 0.80, -0.035, -0.60, 0.0  (tors w_rot=0.02)
		add(1, new gauss(0, 0.8, cutoff));              // WEIGHT: -0.045
		add(1, new repulsion(0.0, cutoff));             // WEIGHT:  0.80
		add(1, new hydrophobic(0.0, 2.5, cutoff));      // WEIGHT: -0.035
		add(1, new non_dir_h_bond(-0.6, 0, cutoff));    // WEIGHT: -0.60
		add(1, new num_tors_div());                     // WEIGHT:  0.0 (w_rot=0.02)
	} else {
		// Default Vina scoring function
		add(1, new gauss(0, 0.5, cutoff));              // WEIGHT: -0.035579
		add(1, new gauss(3, 2.0, cutoff));              // WEIGHT: -0.005156
		add(1, new repulsion(0.0, cutoff));             // WEIGHT:  0.840245
		add(1, new hydrophobic(0.5, 1.5, cutoff));      // WEIGHT: -0.035069
		add(1, new non_dir_h_bond(-0.7, 0, cutoff));    // WEIGHT: -0.587439
		add(1, new num_tors_div());                     // WEIGHT:  1.923
	}
}
