import os
import re
import json

DEFAULT_RECIPES = {
    "reduction": {
        "pattern": r"(sum|reduce|reduction|accumulate|add_kernel)",
        "name": "Shared Memory Reduction & Sequential Addressing",
        "description": "Optimizes cumulative summation kernels using shared memory, sequential addressing to avoid bank conflicts, and loop unrolling.",
        "few_shot_prompt": (
            "OPTIMIZATION RECIPE: Reduction optimization.\n"
            "If you are optimizing a sum/reduction kernel, apply shared memory tiling and sequential addressing instead of interleaved addressing.\n"
            "Example Optimization Pattern:\n"
            "```cuda\n"
            "__global__ void reduce_kernel(const float* g_idata, float* g_odata, unsigned int n) {\n"
            "    extern __shared__ float sdata[];\n"
            "    unsigned int tid = threadIdx.x;\n"
            "    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;\n"
            "    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;\n"
            "    if (i + blockDim.x < n) sdata[tid] += g_idata[i + blockDim.x];\n"
            "    __syncthreads();\n"
            "    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n"
            "        if (tid < s) {\n"
            "            sdata[tid] += sdata[tid + s];\n"
            "        }\n"
            "        __syncthreads();\n"
            "    }\n"
            "    if (tid == 0) g_odata[blockIdx.x] = sdata[0];\n"
            "}\n"
            "```"
        )
    },
    "matmul": {
        "pattern": r"(matmul|matrix_mul|gemm|matrix_multiply)",
        "name": "Shared Memory Tiling for Matrix Multiplication",
        "description": "Optimizes matrix multiplication by loading tiles of matrices into shared memory to reuse data and reduce global memory bandwidth bottleneck.",
        "few_shot_prompt": (
            "OPTIMIZATION RECIPE: Shared Memory Matrix Multiplication Tiling.\n"
            "If you are optimizing matrix multiplication, use 2D shared memory tiles (e.g. 16x16 or 32x32) to cache input blocks and avoid redundant global memory loads.\n"
            "Example Optimization Pattern:\n"
            "```cuda\n"
            "template <int TILE_SIZE>\n"
            "__global__ void matmul_tile_kernel(const float* A, const float* B, float* C, int N) {\n"
            "    __shared__ float s_A[TILE_SIZE][TILE_SIZE];\n"
            "    __shared__ float s_B[TILE_SIZE][TILE_SIZE];\n"
            "    int tx = threadIdx.x; int ty = threadIdx.y;\n"
            "    int col = blockIdx.x * TILE_SIZE + tx;\n"
            "    int row = blockIdx.y * TILE_SIZE + ty;\n"
            "    float sum = 0.0f;\n"
            "    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {\n"
            "        if (row < N && m * TILE_SIZE + tx < N) s_A[ty][tx] = A[row * N + m * TILE_SIZE + tx];\n"
            "        else s_A[ty][tx] = 0.0f;\n"
            "        if (col < N && m * TILE_SIZE + ty < N) s_B[ty][tx] = B[(m * TILE_SIZE + ty) * N + col];\n"
            "        else s_B[ty][tx] = 0.0f;\n"
            "        __syncthreads();\n"
            "        for (int k = 0; k < TILE_SIZE; ++k) sum += s_A[ty][k] * s_B[k][tx];\n"
            "        __syncthreads();\n"
            "    }\n"
            "    if (row < N && col < N) C[row * N + col] = sum;\n"
            "}\n"
            "```"
        )
    },
    "vector_add": {
        "pattern": r"(vectoradd|vector_add|add_arrays|elementwise)",
        "name": "Grid-Stride Loops & Memory Coalescing",
        "description": "Optimizes 1D elementwise operations using Grid-Stride Loops to make the kernel size-independent and ensure fully coalesced global memory access patterns.",
        "few_shot_prompt": (
            "OPTIMIZATION RECIPE: Grid-Stride Elements Loops.\n"
            "Ensure elements are accessed via stride indexes (`i += blockDim.x * gridDim.x`) to support large data sizes and align memory coalescing.\n"
            "Example Optimization Pattern:\n"
            "```cuda\n"
            "__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {\n"
            "    int index = blockDim.x * blockIdx.x + threadIdx.x;\n"
            "    int stride = blockDim.x * gridDim.x;\n"
            "    for (int i = index; i < n; i += stride) {\n"
            "        C[i] = A[i] + B[i];\n"
            "    }\n"
            "}\n"
            "```"
        )
    }
}

class RecipeDatabase:
    def __init__(self, recipe_file=None):
        if not recipe_file:
            local_recipe = os.path.join(os.getcwd(), ".cudallm-recipes.json")
            if os.path.exists(local_recipe):
                self.recipe_file = local_recipe
            else:
                app_dir = os.path.expanduser("~/.cudallm")
                os.makedirs(app_dir, exist_ok=True)
                self.recipe_file = os.path.join(app_dir, "optimization_recipes.json")
        else:
            self.recipe_file = recipe_file

        self.recipes = dict(DEFAULT_RECIPES)
        self.load_recipes()

    def load_recipes(self):
        if os.path.exists(self.recipe_file):
            try:
                with open(self.recipe_file, 'r', encoding='utf-8') as f:
                    custom_recipes = json.load(f)
                    self.recipes.update(custom_recipes)
            except Exception:
                pass

    def save_recipes(self):
        try:
            with open(self.recipe_file, 'w', encoding='utf-8') as f:
                json.dump(self.recipes, f, indent=2)
        except Exception:
            pass

    def match_recipe(self, code):
        for recipe_id, recipe in self.recipes.items():
            pattern = recipe.get("pattern", "")
            if pattern and re.search(pattern, code, re.IGNORECASE):
                return recipe
        return None

    def add_custom_recipe(self, name, pattern, prompt, description=""):
        recipe_id = re.sub(r'[^a-z0-9]', '_', name.lower())
        self.recipes[recipe_id] = {
            "name": name,
            "pattern": pattern,
            "few_shot_prompt": prompt,
            "description": description
        }
        self.save_recipes()
