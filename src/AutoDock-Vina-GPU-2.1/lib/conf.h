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

#ifndef VINA_CONF_H
#define VINA_CONF_H

#include <boost/ptr_container/ptr_vector.hpp> // typedef output_container

#include "quaternion.h"
#include "random.h"

/* ��������仯�е�λ�á�����Ť�ر仯
	position��orientation��torsion:double
	position=position_��orientation=orientation_��torsion=torsion_
*/
struct scale {
	fl position;
	fl orientation;
	fl torsion;
	scale(fl position_, fl orientation_, fl torsion_) : position(position_), orientation(orientation_), torsion(torsion_) {}
};



/*
	ligands��flex��unint������

*/
struct conf_size {
	szv ligands;
	szv flex;
	/*
		sum���������ǽ������е�Ԫ�ؽ����ۼӣ�
		demo��vertor<int> ligands=[1,1,1,1];
			  vertor<int> flex=[2,2,2,2];
			  ����4+8+4=16��
		
	*/
	sz num_degrees_of_freedom() const {
		return sum(ligands) + sum(flex) + 6 * ligands.size();
	}
};



/*	
	input����torsions��double������
	�ú����ǽ�torsions����Ԫ�ض���ֵΪ0
*/
inline void torsions_set_to_null(flv& torsions) {
	VINA_FOR_IN(i, torsions)               //for(i=0;i<torsions.size;i++)
		torsions[i] = 0;
}

/*	��torsions�������б�׼��
	input����torsions��double������c��double������factor��double��

*/
inline void torsions_increment(flv& torsions, const flv& c, fl factor) { // new torsions are normalized
	VINA_FOR_IN(i, torsions) {                            //for(i=0;i<torsions.size;i++)
		torsions[i] += normalized_angle(factor * c[i]);   //normalized_angle�����ԽǶȽ��б�׼������ΧΪ[-pi,pi];
		normalize_angle(torsions[i]);                     //normalized_angle�����ԽǶȽ��б�׼������ΧΪ[-pi,pi];
	}
}

/*	
    �ú�������[-pi,pi]���������torsions��Ť�أ�
	input����torsions��double������generator�������������
*/
inline void torsions_randomize(flv& torsions, rng& generator) {
	VINA_FOR_IN(i, torsions)                        //for(i=0;i<torsions.size;i++)
		torsions[i] = random_fl(-pi, pi, generator);//���������
}

/*	�ж�torsions1��torsions2�Ƿ����ƣ�Ť��ʮ�����ƣ�
	input����torsions1��torsions2��double������cutoff��double��
	output����bool����
	��torsions1��torsions2����Ԫ�ظ�����ȵ�����£����ݣ�torsions1[i] - torsions2[i]���ĽǶȱ�׼��
	�ľ���ֵ�Ƿ��������cutoff���ǵĻ�����false������true
*/
inline bool torsions_too_close(const flv& torsions1, const flv& torsions2, fl cutoff) {
	assert(torsions1.size() == torsions2.size());
	VINA_FOR_IN(i, torsions1)                           //for(i=0;i<torsions1.size;i++)
		if(std::abs(normalized_angle(torsions1[i] - torsions2[i])) > cutoff) 
			return false;
	return true;
}

/*	��������Ť��
	input����torsions��double������spread��rp��double��rs��double����ָ�룬generator�������������
	����Ҫ�ж�rs�Ƿ��ָ�����rsָ�������Ƿ���torsions����Ԫ�ظ�����ȣ��ǵĻ�����������ֹ��
	if��rs��Ϊ��ָ�벢������[0,1]�����С��rp��
		��rs������ֵ��torsions������
	else
		torsions[i]=torsions[i]+����[-spread, spread]�����	
*/
inline void torsions_generate(flv& torsions, fl spread, fl rp, const flv* rs, rng& generator) {
	assert(!rs || rs->size() == torsions.size());                  //if present, rs should be the same size as torsions
	VINA_FOR_IN(i, torsions)                                       //for(i=0;i<torsions1.size;i++)
		if(rs && random_fl(0, 1, generator) < rp)                  //random_fl����[0,1]�����
			torsions[i] = (*rs)[i];
		else
			torsions[i] += random_fl(-spread, spread, generator); //random_fl����[-spread, spread]�����
}


/*
	����仯
*/
struct rigid_change {
	vec position;                //vec�ṹ�壬���ڴ����Ԫ���鲿
	vec orientation;		     //vec�ṹ�壬���ڴ����Ԫ���鲿
	/*   ������ʼ��
	   position�ṹ���У� data[0] = 0��data[1] = 0;data[2] = 0;
	   orientation�ṹ���У� data[0] = 0��data[1] = 0;data[2] = 0;
	*/
	rigid_change() : position(0, 0, 0), orientation(0, 0, 0) {}
	/*
	   ��һ��ӡposition��orientation������,���ԡ�,���ָ�
	*/
	void print() const {
		::print(position);   //��һ��ӡposition������,���ԡ�,���ָ�
		::print(orientation);
	}
};

/* 
	��������
*/
struct rigid_conf {
	vec position;                    //vec�ṹ�壬���ڴ����Ԫ���鲿
	qt orientation;					 //��Ԫ��
	/*   ������ʼ��
	   position�ṹ���У� data[0] = 0��data[1] = 0;data[2] = 0;
	   orientation��һ��ʵ��Ϊ1���鲿��Ϊ0����Ԫ��
	*/
	rigid_conf() : position(0, 0, 0), orientation(qt_identity) {}

	/*   ������Ϊnull
		vec�ṹ���У� data[0] = 0��data[1] = 0;data[2] = 0;
		orientation��һ��ʵ��Ϊ1���鲿��Ϊ0����Ԫ��
	*/
	void set_to_null() {
		position = zero_vec;
		orientation = qt_identity;
	}

/*
	input����c��rigid_change�ṹ�壬factor��double��
	inter����rotation��vec �ṹ�壬orientation����Ԫ����
	c.position��c.orientation:vec�ṹ�壻
	position=(position.data[0]+factor * c.position.data[0],position.data[1]+factor * c.position.data[1], position.data[2]+factor * c.position.data[2])
	rotation=��factor * c.orientation.data[0]��factor * c.orientation.data[1]��factor * c.orientation.data[2]��
	quaternion_increment(orientation, rotation)��Ԫ���˷��Լ����Ʊ�׼����
	orientation��w1,x1,y1,z1��;rotation��w2,x2,y2,z2��
	orientation * rotation =
	(w1*w2 - x1*x2 - y1*y2 - z1*z2) +
	(w1*x2 + x1*w2 + y1*z2 - z1*y2) i +
	(w1*y2 - x1*z2 + y1*w2 + z1*x2) j +
	(w1*z2 + x1*y2 - y1*x2 + z1*w2) k
*/
	void increment(const rigid_change& c, fl factor) {
		position += factor * c.position;       //factor * data[0], factor * data[1], factor * data[2]
		                                       //ֵ���δ���vec�ṹ���data[0]~data[2]
		vec rotation;                          //vec�ṹ�壬���ڴ����Ԫ���鲿
		rotation = factor * c.orientation;
		quaternion_increment(orientation, rotation); // orientation does not get normalized; tests show rounding errors growing very slowly
	}

/*�����������ı仯λ�á�����
	input����corner1��corner2��vec�ṹ�壬generator�������������
*/
	void randomize(const vec& corner1, const vec& corner2, rng& generator) {
		position = random_in_box(corner1, corner2, generator);//��������vec�ṹ�壬����һ��[corner1[i], corner2[i]]���vec�ṹ��
		orientation = random_orientation(generator);
	}

	/*
		input����c��rigid_conf�ṹ�壬position_cutoff��orientation_cutoff��double��
		output����bool��
		if  vec_distance_sqr(position, c.position)����������vec�����data����Ԫ�ز��ƽ������
		sqr(position_cutoff)=position_cutoff*position_cutoff������false��
		if  ����quaternion_difference(orientation, c.orientation)���ص�vec�����data����Ԫ�ص�ƽ���� ����
		orientation_cutoff*orientation_cutoff������false��
		else
		����true��
	*/
	bool too_close(const rigid_conf& c, fl position_cutoff, fl orientation_cutoff) const {
		if(vec_distance_sqr(position, c.position) > sqr(position_cutoff)) return false;
		if(sqr(quaternion_difference(orientation, c.orientation)) > sqr(orientation_cutoff)) return false;
		return true;
	}

	/*	����λ�ñ仯
		input����spread��double��generator���������������
		position��vec�ṹ�壬���ڴ����Ԫ���鲿
		random_inside_sphere(generator)���ڵ�λ�����������vec�ṹ��tmp
	position=(position.data[0]+spread * tmp.data[0],position.data[1]+spread * tmp.data[1], position.data[2]+spread * tmp.data[2])
	*/
	void mutate_position(fl spread, rng& generator) {
		position += spread * random_inside_sphere(generator);//�ڵ�λ�����������vec�ṹ��
	}

/*    ���巽��仯
	input����spread��double��generator���������������
	inter����orientation����Ԫ��
	tmp��vec�ṹ�壬���ڴ����Ԫ���鲿
	random_inside_sphere(generator)���ڵ�λ�����������vec�ṹ��tmp�����tmp�ṹ�����������е�tmp����һ��
	tmp=(spread * tmp.data[0],spread * tmp.data[1],spread * tmp.data[2])��
	������������Ԫ���˷��Լ����Ʊ�׼����quaternion_increment(orientation, tmp)
*/
	void mutate_orientation(fl spread, rng& generator) {
		vec tmp;
		tmp = spread * random_inside_sphere(generator);
		quaternion_increment(orientation, tmp);
	}


/*  ��������仯��λ�á�����Ť��
	input����position_spread��orientation_spread��rp��double��rs��rigid_conf�ṹ��ָ�룬generator���������������
	inter���� position��vec�ṹ��,orientation:��Ԫ��  
	if rs��Ϊ��ָ�벢����[0,1]֮�����ɵ�double�����С��rp
		position=rs�ṹ���е�position��
		orientation=rs�ṹ���е�orientation��
	else
		position=(position.data[0]+position_spread * tmp.data[0],position.data[1]+position_spread * tmp.data[1], position.data[2]+position_spread * tmp.data[2])��
		quaternion_increment(orientation, (orientation_spread * tmp.data[0],orientation_spread * tmp.data[1],orientation_spread * tmp.data[2]))��
		����������Ԫ�����г˷��Լ����Ʊ�׼����
		tmp���ڵ�λ�����������vec�ṹ�壻
*/
	void generate(fl position_spread, fl orientation_spread, fl rp, const rigid_conf* rs, rng& generator) {
		if(rs && random_fl(0, 1, generator) < rp)
			position = rs->position;
		else
			mutate_position(position_spread, generator);
		if(rs && random_fl(0, 1, generator) < rp)
			orientation = rs->orientation;
		else
			mutate_orientation(orientation_spread, generator);
	}
	
/*
	input����in��out��vec�ṹ��������begin��end��unint
	inter����m��mat�ṹ��,position:vec�ṹ��
	orientation��a,b,c,d��----------->
	aa = a*a;ab = a*b;ac = a*c;ad = a*d;bb = b*b
	bc = b*c;bd = b*d;cc = c*c;cd = c*d;dd = d*d 
	----------->m�ṹ����data
	data[0]=(aa + bb - cc - dd)��data[3]=2 * (-ad + bc)     ��data[6]=2 * (ac + bd)      ;
	data[1]=( 2 * (ad + bc)    ��data[4]=(aa - bb + cc - dd)��data[7]= 2 * (-ab + cd)    ;
	data[2]=2 * (-ac + bd)     ��data[5]=2 * (ab + cd)      ��data[8]=(aa - bb - cc + dd);
	----------->out[i]�ṹ����data
	(m.data[0]*in[i][0] + m.data[3]*in[i][1] + m.data[6]*in[i][2]+position.data[0],
	 m.data[1]*in[i][0] + m.data[4]*in[i][1] + m.data[7]*in[i][2]+position.data[1],
	 m.data[2]*in[i][0] + m.data[5]*in[i][1] + m.data[8]*in[i][2]+position.data[2])
*/	
	void apply(const vecv& in, vecv& out, sz begin, sz end) const {
		assert(in.size() == out.size());   
		const mat m = quaternion_to_r3(orientation);
		VINA_RANGE(i, begin, end)                     //for(i=begin;i<end;i++)
			out[i] = m * in[i] + position;
	}

/*  ��ӡ�����λ�á�����
	��һ��ӡvec position������,���ԡ�,���ָ�
	��ӡ��orientation��Ԫ��
*/
	void print() const {
		::print(position);
		::print(orientation);
	}

private:
	friend class boost::serialization::access; //��Ԫ����
	template<class Archive>
	void serialize(Archive & ar, const unsigned version) {
		ar & position;
		ar & orientation;
	}
};

/* 
	�ı�����
*/
struct ligand_change {
	rigid_change rigid;      //rigid_change�ṹ��
	flv torsions;            //double����                
	/*
	��һ��ӡvec�ṹ��position��orientation������,���ԡ�,���ָ�
	�Լ���ӡtorsions����
	*/
	void print() const {
		rigid.print();
		printnl(torsions);
	}
};

/*
	��������
*/
struct ligand_conf {
	rigid_conf rigid;		//rigid_change�ṹ��
	flv torsions;           //double����  
	/*   ������Ϊnull
		rigid�ṹ���У�
		data[0] = 0��data[1] = 0;data[2] = 0;
		orientation��һ��ʵ��Ϊ1���鲿��Ϊ0����Ԫ��
		torsions����Ԫ�ض���ֵΪ0
	*/
	void set_to_null() {
		rigid.set_to_null();
		torsions_set_to_null(torsions);
	}

	/*
		input����c��rigid��ligand_change�ṹ�壬factor��double��c.rigid��rigid_change�ṹ�壻
		c.rigid.position��c.rigid.orientation:vec�ṹ�壻
	  ��rigid_conf�ṹ���У�position=(position.data[0]+factor *c.rigid.position.data[0],position.data[1]+factor * c.rigid.position.data[1], position.data[2]+factor * c.rigid.position.data[2])
		rotation=��factor * c.rigid.orientation.data[0]��factor * c.rigid.orientation.data[1]��factor * c.rigid.orientation.data[2]��
		quaternion_increment(orientation��rigid_conf�ṹ���У�, rotation)��Ԫ���˷��Լ����Ʊ�׼����
		orientation��w1,x1,y1,z1��;rotation��w2,x2,y2,z2��
		orientation * rotation =
		(w1*w2 - x1*x2 - y1*y2 - z1*z2) +
		(w1*x2 + x1*w2 + y1*z2 - z1*y2) i +
		(w1*y2 - x1*z2 + y1*w2 + z1*x2) j +
		(w1*z2 + x1*y2 - y1*x2 + z1*w2) k
		����torsions�������б�׼��
	*/
	void increment(const ligand_change& c, fl factor) {
		rigid.increment(c.rigid, factor);   //c.rigid��rigid_change�ṹ��
		torsions_increment(torsions, c.torsions, factor);//��torsions�������б�׼��
	}

	/*	���������position�ṹ���orientation��Ԫ���Լ�[-pi,pi]���������torsions
		input����rigid��ligand_change�ṹ�壬corner1��corner2��vec�ṹ�壬generator�������������
		inter����torsions��double������rigid��rigid_change�ṹ�壻
		
	*/
	void randomize(const vec& corner1, const vec& corner2, rng& generator) {
		rigid.randomize(corner1, corner2, generator);
		torsions_randomize(torsions, generator);
	}
	/* ��ӡ���Ա仯�е�λ�á�����Ť����Ϣ
		inter����rigid��rigid_change�ṹ�壻
		��һ��ӡvec position������,���ԡ�,���ָ�ʹ�ӡ��orientation��Ԫ���Լ���ӡtorsions����
	*/
	void print() const {
		rigid.print();
		printnl(torsions);
	}
private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned version) {
		ar & rigid;
		ar & torsions;
	}
};

/*   �л��仯
	torsions��double ����
*/
struct residue_change {
	flv torsions;
	/*
	��ӡtorsions����
	*/
	void print() const {
		printnl(torsions);
	}
};

/*   �л�����
	torsions��double ����

*/
struct residue_conf {
	flv torsions;

	/*
	input����torsions��double������
	�ú����ǽ�torsions����Ԫ�ض���ֵΪ0
	*/
	void set_to_null() {
		torsions_set_to_null(torsions);
	}


	/*	��torsions�������б�׼��
	input����c��residue_change�ṹ�壻factor��double��
	*/
	void increment(const residue_change& c, fl factor) {
		torsions_increment(torsions, c.torsions, factor);
	}

	/*�ú�������[-pi,pi]���������torsions��Ť�أ�
	 input����generator�������������
	*/
	void randomize(rng& generator) {
		torsions_randomize(torsions, generator);
	}

	/*
		��ӡŤ����Ϣ
	*/
	void print() const {
		printnl(torsions);
	}
private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned version) {
		ar & torsions;
	}
};

/*   
	s:conf_size�ṹ��
	ligands��ligand_change�ṹ��������
	flex��residue_change�ṹ��������
	ligands������Ԫ�ظ���Ϊconf_size�ṹ���е�ligands uint������Ԫ�ظ���;
	flex������Ԫ�ظ���Ϊconf_size�ṹ���е�flex uint������Ԫ�ظ���;

	ligands[i].torsions double������Ԫ�ظ���Ϊs.ligands[ligands.size]������ֵ��Ϊ0��
	flex[i].torsions double������Ԫ�ظ���Ϊs.flex[flex.size]������ֵ��Ϊ0��

*/
struct change {
	std::vector<ligand_change> ligands;    //ligand_change�ṹ������
	std::vector<residue_change> flex;
	change(const conf_size& s) : ligands(s.ligands.size()), flex(s.flex.size()) {

		VINA_FOR_IN(i, ligands)     //for (i = 0; i<ligands.size; i++)
			ligands[i].torsions.resize(s.ligands[i], 0);
		VINA_FOR_IN(i, flex)	  //for (i = 0; i<flex.size; i++)
			flex[i].torsions.resize(s.flex[i], 0);
	}

	/*
		����change��sz�������ض���
		input����index��unint ��

		ligands��ligand_change�ṹ��������
		flex��residue_change�ṹ��������
		ligΪligands[i]�ṹ��ı������ı�lig���Ǹı�ligands[i]��
		resΪflex[i]�ṹ��ı������ı�res���Ǹı�flex[i]


		for (i = 0; i<ligands.size; i++)
		if��index<3��
			return      rigid_change�ṹ����position�ṹ��ĵ�index��Ԫ��
		index=index-3��������ifû��ִ�о�һ����ִ�и���䣬ע�������indexΪ�޷������ͣ�
		if��index<3��
		return      rigid_change�ṹ����orientation�ṹ��ĵ�index��Ԫ��
		index=index-3��������ifû��ִ�о�һ����ִ�и���䣬ע�������indexΪ�޷������ͣ�
		if��ligands[i]�ṹ����torsions������Ԫ�ظ���<3��
		return      ligand_change�ṹ����torsions�����ĵ�index��Ԫ��
		index=ligands[i]�ṹ����torsions������Ԫ�ظ���-3��������ifû��ִ�о�һ����ִ�и���� ��ע�������indexΪ�޷������ͣ�
	
	    for (i = 0; i<flex.size; i++)
		if��flex[i]�ṹ����torsions������Ԫ�ظ���<3��
		return      residue_change�ṹ����torsions�����ĵ�index��Ԫ��
		index=flex[i]�ṹ����torsions������Ԫ�ظ���-3��������ifû��ִ�о�һ����ִ�и���� ��ע�������indexΪ�޷������ͣ�
		�����return��ȴֻ��ִ��һ��
	*/

	fl operator()(sz index) const { // returns by value
		VINA_FOR_IN(i, ligands) {  //for (i = 0; i<ligands.size; i++)
			const ligand_change& lig = ligands[i];
			if(index < 3) return lig.rigid.position[index];
			index -= 3;
			if(index < 3) return lig.rigid.orientation[index];
			index -= 3;
			if(index < lig.torsions.size()) return lig.torsions[index];
			index -= lig.torsions.size();
		}
		VINA_FOR_IN(i, flex) {    //for (i = 0; i<flex.size; i++)
			const residue_change& res = flex[i];
			if(index < res.torsions.size()) return res.torsions[index];
			index -= res.torsions.size();
		}
		VINA_CHECK(false); 
		return 0; // shouldn't happen, placating the compiler
	}

/* ����change��sz������������
	���������غ���ʵ�ֵ�����һ�£�ֻ�Ƿ��ز�ͬ��
*/
	fl& operator()(sz index) {
		VINA_FOR_IN(i, ligands) {
			ligand_change& lig = ligands[i];
			if(index < 3) return lig.rigid.position[index];
			index -= 3;
			if(index < 3) return lig.rigid.orientation[index];
			index -= 3;
			if(index < lig.torsions.size()) return lig.torsions[index];
			index -= lig.torsions.size();
		}
		VINA_FOR_IN(i, flex) {
			residue_change& res = flex[i];
			if(index < res.torsions.size()) return res.torsions[index];
			index -= res.torsions.size();
		}
		VINA_CHECK(false); 
		return ligands[0].rigid.position[0]; // shouldn't happen, placating the compiler
	}


/*	����ligand_change��residue_change�ṹ��������torsionsԪ�ظ����ܺͼ���6*ligands.size���ܺ�
	ligands��ligand_change�ṹ��������
	flex��residue_change�ṹ��������
*/
	sz num_floats() const {
		sz tmp = 0;
		VINA_FOR_IN(i, ligands)                       //for (i = 0; i<ligands.size; i++)
			tmp += 6 + ligands[i].torsions.size();
		VINA_FOR_IN(i, flex)                          //for (i = 0; i<flex.size; i++)
			tmp += flex[i].torsions.size();
		return tmp;
	}

	/*
	��һ��ӡligands�ṹ������������vec�ṹ��position��orientation������,���ԡ�,���ָ�
	�Լ���ӡ���е�torsions������
	��ӡflex�ṹ���������е�torsions������
	ligands��ligand_change�ṹ��������
	flex��residue_change�ṹ��������
	
	*/

	void print() const {
		VINA_FOR_IN(i, ligands)
			ligands[i].print();
		VINA_FOR_IN(i, flex)
			flex[i].print();
	}
};

/*  ����
		ligands��ligand_conf�ṹ����������������
	    flex��residue_conf�ṹ��������  ��������
		ligands������Ԫ�ظ���Ϊconf_size�ṹ���е�ligands unint������Ԫ�ظ���;
		flex������Ԫ�ظ���Ϊconf_size�ṹ���е�flex uint������Ԫ�ظ���;

		ligands[i].torsions double������Ԫ�ظ���Ϊs.ligands[ligands.size]������ֵ��Ϊ0��
		flex[i].torsions double������Ԫ�ظ���Ϊs.flex[flex.size]������ֵ��Ϊ0��
*/
struct conf {
	std::vector<ligand_conf> ligands;
	std::vector<residue_conf> flex;
	conf() {}
	conf(const conf_size& s) : ligands(s.ligands.size()), flex(s.flex.size()) {
		VINA_FOR_IN(i, ligands)
			ligands[i].torsions.resize(s.ligands[i], 0); // FIXME?
		VINA_FOR_IN(i, flex)
			flex[i].torsions.resize(s.flex[i], 0); // FIXME?
	}

	/*  ������Ϊnull
		ligands�ṹ�������У�
		data[0] = 0��data[1] = 0;data[2] = 0;
		orientation��һ��ʵ��Ϊ1���鲿��Ϊ0����Ԫ��
		torsions����Ԫ�ض���ֵΪ0��
		flex�ṹ�������У�
		torsions����Ԫ�ض���ֵΪ0
	*/
	void set_to_null() {
		VINA_FOR_IN(i, ligands)
			ligands[i].set_to_null();
		VINA_FOR_IN(i, flex)
			flex[i].set_to_null();
	}

	/*	������Ա仯��λ�á�����Ť���Լ�������Ť�ؽ��б�׼��
		����c.ligands[i]�ṹ���е�position��orientation �ṹ���Լ���ligand_change�ṹ����torsions�������б�׼��
	�Լ���flex�ṹ�������е�torsions�������б�׼��
    */
	void increment(const change& c, fl factor) { // torsions get normalized, orientations do not
		VINA_FOR_IN(i, ligands)
			ligands[i].increment(c.ligands[i], factor);
		VINA_FOR_IN(i, flex)
			flex[i]   .increment(c.flex[i],    factor);
	}
  /* �ڲ�����仯�Ƿ�����
    �ж�ligands�ṹ�������е�torsions�Ƿ��conf.�ṹ�������е�torsions����
	�ǵĻ�����true�����򷵻�false
	input����c��conf�ṹ�壬torsions_cutoff��double
	ligands��ligand_conf�ṹ��������
	flex��residue_conf�ṹ��������
   */
	bool internal_too_close(const conf& c, fl torsions_cutoff) const {
		assert(ligands.size() == c.ligands.size());               //�ж�ligands�ṹ��������Ԫ�ظ����Ƿ��conf�ṹ���е�ligands�ṹ��������Ԫ�ظ������
		VINA_FOR_IN(i, ligands)                                   //for (i = 0; i<ligands.size; i++)
			if(!torsions_too_close(ligands[i].torsions, c.ligands[i].torsions, torsions_cutoff))
				return false;   //�ж�torsions1��torsions2�Ƿ�����
		return true;
	}
/*	�ⲿ����仯�Ƿ�����
	input����c��conf�ṹ�壻cutoff��scale�ṹ��
	���ж�ligands[i].rigid�е�position��orientation�Ƿ��c.ligands[i].rigid�е�position��orientation���ƣ����ǵĻ�����false��
	�����ж�flex[i]�е�torsions������c.flex[i]�е�torsions�����Ƿ����ƣ����ǵĻ�����false��
	Ҫ����󷵻�true
*/
	bool external_too_close(const conf& c, const scale& cutoff) const {
		assert(ligands.size() == c.ligands.size());
		VINA_FOR_IN(i, ligands)
			if(!ligands[i].rigid.too_close(c.ligands[i].rigid, cutoff.position, cutoff.orientation))
				return false;
		assert(flex.size() == c.flex.size());
		VINA_FOR_IN(i, flex)
			if(!torsions_too_close(flex[i].torsions, c.flex[i].torsions, cutoff.torsion))
				return false;
		return true;
	}
/*	�ⲿ������ڲ�����仯�Ƿ�����
	input����c��conf�ṹ�壻cutoff��scale�ṹ��;
	

*/
	bool too_close(const conf& c, const scale& cutoff) const {
		return internal_too_close(c, cutoff.torsion) &&
			   external_too_close(c, cutoff); // a more efficient implementation is possible, probably
	}

/*  ���������ڲ��仯��λ�á�����Ť�ء�
	input����torsion_spread��rp��double��rs:conf�ṹ��ָ�룬generator�������������
    λ�ã� position��0��0��0����
	���� orientation��1��0��0��0����
	Ť�أ�ligands[i].torsions=ligands[i].torsions+����[-torsion_spread, torsion_spread]�����
*/
	void generate_internal(fl torsion_spread, fl rp, const conf* rs, rng& generator) { // torsions are not normalized after this
		VINA_FOR_IN(i, ligands) {
			ligands[i].rigid.position.assign(0);
			ligands[i].rigid.orientation = qt_identity;
			const flv* torsions_rs = rs ? (&rs->ligands[i].torsions) : NULL;//rs�ṹ��ָ�벿Ϊ��ʱ��ֵtorsions����
			torsions_generate(ligands[i].torsions, torsion_spread, rp, torsions_rs, generator);
		}
	}


   /*  �����ⲿ�仯�����ԣ�λ�á�����Ť�أ������ԣ�Ť�أ���
   input����spread��scale�ṹ�壬rp��double��rs:conf�ṹ��ָ�룬generator�������������
	*/

	void generate_external(const scale& spread, fl rp, const conf* rs, rng& generator) { // torsions are not normalized after this
		VINA_FOR_IN(i, ligands) {
			const rigid_conf* rigid_conf_rs = rs ? (&rs->ligands[i].rigid) : NULL;
			ligands[i].rigid.generate(spread.position, spread.orientation, rp, rigid_conf_rs, generator);
		}
		VINA_FOR_IN(i, flex) {
			const flv* torsions_rs = rs ? (&rs->flex[i].torsions) : NULL;
			torsions_generate(flex[i].torsions, spread.torsion, rp, torsions_rs, generator);
		}
	}

	/*���ɸ��Ե����λ�á�����Ť�غ������е����Ť�أ�
	ligands�ṹ�����������������position�ṹ���orientation��Ԫ���Լ�[-pi,pi]���������torsions��
	flex�ṹ������������[-pi,pi]���������torsions
	*/
	void randomize(const vec& corner1, const vec& corner2, rng& generator) {
		VINA_FOR_IN(i, ligands)
			ligands[i].randomize(corner1, corner2, generator);
		VINA_FOR_IN(i, flex)
			flex[i].randomize(generator);
	}

	/*
	 ��ӡ���Ա仯�е�λ�á�����Ť����Ϣ�Լ������е�Ť����Ϣ
	*/
	void print() const {
		VINA_FOR_IN(i, ligands)
			ligands[i].print();
		VINA_FOR_IN(i, flex)
			flex[i].print();
	}
private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned version) {
		ar & ligands;
		ar & flex;
	}
};


/* �������
     c����conf �ṹ�壬������      e����double��   coords����vec�ṹ������   
	 ��ʼ��

 */
struct output_type {
	conf c;
	fl e;
	fl e_gpu;  // GPU-scored total energy (Vina+QFD); 0 if from CPU path
	fl e_ls;   // LS metal coordination bonus (negative = good; added to e during sort)
	vecv coords;
	output_type(const conf& c_, fl e_) : c(c_), e(e_), e_gpu(0), e_ls(0) {}
};

typedef boost::ptr_vector<output_type> output_container;   //output_type�ṹ������


/*
	����<
*/
inline bool operator<(const output_type& a, const output_type& b) { // for sorting output_container
	return a.e < b.e;
}

#endif
