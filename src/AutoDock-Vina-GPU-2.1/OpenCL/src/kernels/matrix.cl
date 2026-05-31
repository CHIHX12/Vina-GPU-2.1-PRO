// symmetric matrix (only half of it are stored)
typedef struct {
	float data[MAX_HESSIAN_MATRIX_SIZE];
	int dim;
}matrix;

void matrix_init(matrix* m, int dim, float fill_data) {
	m->dim = dim;
	// (size assert removed)
	//((dim * (dim + 1) / 2)*sizeof(float)); // symmetric matrix
	for (int i = 0; i < (dim * (dim + 1) / 2); i++)m->data[i] = fill_data;
	for (int i = (dim * (dim + 1) / 2); i < MAX_HESSIAN_MATRIX_SIZE; i++)m->data[i] = 0;// Others will be 0
}

// as rugular 3x3 matrix
void mat_init(matrix* m, float fill_data) {
	m->dim = 3; // fixed to 3x3 matrix
	// (assert removed)
	for (int i = 0; i < 9; i++)m->data[i] = fill_data;
}


void matrix_set_diagonal(matrix* m, float fill_data) {
	for (int i = 0; i < m->dim; i++) {
		m->data[i + i * (i + 1) / 2] = fill_data;
	}
}

// as rugular matrix
inline void matrix_set_element(matrix* m, int dim, int x, int y, float fill_data) {
	m->data[x + y * dim] = fill_data;
}

inline void matrix_set_element_tri(matrix* m, int x, int y, float fill_data) {
	m->data[x + y*(y+1)/2] = fill_data;
}
inline int tri_index(int n, int i, int j) {
	// (assert removed)
	return i + j * (j + 1) / 2;
}

inline int index_permissive(const matrix* m, int i, int j) {
	return (i < j) ? tri_index(m->dim, i, j) : tri_index(m->dim, j, i);
}