import re
import os
import csv

class RooflineAnalyzer:
    TYPE_SIZE_BYTES = {
        "half": 2,
        "__half": 2,
        "char": 1,
        "signed char": 1,
        "unsigned char": 1,
        "bool": 1,
        "short": 2,
        "short int": 2,
        "unsigned short": 2,
        "unsigned short int": 2,
        "int": 4,
        "unsigned": 4,
        "unsigned int": 4,
        "long": 8,
        "long int": 8,
        "unsigned long": 8,
        "unsigned long int": 8,
        "long long": 8,
        "long long int": 8,
        "unsigned long long": 8,
        "unsigned long long int": 8,
        "float": 4,
        "double": 8,
    }

    FUNC_FLOP_WEIGHTS = {
        "sin": 4,
        "cos": 4,
        "sqrt": 4,
        "exp": 8,
        "log": 8,
        "pow": 10,
        "fma": 2,
        "fmaf": 2,
    }

    @staticmethod
    def _sanitize_kernel_source(code):
    
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'"(?:\\.|[^"\\])*"', '""', code)
        code = re.sub(r"'(?:\\.|[^'\\])*'", "''", code)
        return code

    @staticmethod
    def _extract_global_kernels(code):
        kernels = []
        pattern = re.compile(r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)\s*\{', re.MULTILINE)
        for match in pattern.finditer(code):
            kernel_name = match.group(1)
            params = match.group(2)
            brace_start = match.end() - 1
            depth = 0
            end_idx = brace_start
            while end_idx < len(code):
                ch = code[end_idx]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        break
                end_idx += 1

            if depth == 0 and end_idx > brace_start:
                body = code[brace_start + 1:end_idx]
                kernels.append({
                    "name": kernel_name,
                    "params": params,
                    "body": body,
                })
        return kernels

    @staticmethod
    def _parse_pointer_params(param_text):
        ptrs = {}
        for raw_param in [p.strip() for p in param_text.split(',') if p.strip()]:
            if '*' not in raw_param:
                continue

            token = raw_param.split('=')[0].strip()
          
            name_match = re.search(r'([A-Za-z_]\w*)\s*$', token)
            if not name_match:
                continue
            var_name = name_match.group(1)

            base_decl = token[:name_match.start()]
            base_decl = base_decl.replace('*', ' ').replace('&', ' ')
            base_decl = re.sub(r'\b(const|volatile|__restrict__|restrict)\b', ' ', base_decl)
            base_decl = re.sub(r'\s+', ' ', base_decl).strip().lower()
            if not base_decl:
                base_decl = 'float'

            size = RooflineAnalyzer.TYPE_SIZE_BYTES.get(base_decl)
            if size is None:
             
                if base_decl.endswith('64_t'):
                    size = 8
                elif base_decl.endswith('16_t'):
                    size = 2
                elif base_decl.endswith('8_t'):
                    size = 1
                else:
                    size = 4
            ptrs[var_name] = {
                "size": size,
                "base_type": base_decl,
                "is_float": base_decl in {"half", "__half", "float", "double"},
            }
        return ptrs

    @staticmethod
    def _extract_float_symbols(param_text, kernel_body):
        float_symbols = set()

        pointer_info = RooflineAnalyzer._parse_pointer_params(param_text)
        for name, info in pointer_info.items():
            if info.get("is_float"):
                float_symbols.add(name)

        decl_pattern = re.compile(
            r'\b(?:const\s+)?(?:half|__half|float|double)\b\s+([^;]+);',
            re.IGNORECASE,
        )
        for m in decl_pattern.finditer(kernel_body):
            decls = m.group(1)
            for part in decls.split(','):
                part = part.strip()
                if not part:
                    continue
                name_match = re.search(r'([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*$', part)
                if name_match:
                    float_symbols.add(name_match.group(1))

      
        assign_pattern = re.compile(
            r'\b(?:half|__half|float|double)\b\s+([A-Za-z_]\w*)\s*=',
            re.IGNORECASE,
        )
        for m in assign_pattern.finditer(kernel_body):
            float_symbols.add(m.group(1))

        return float_symbols

    @staticmethod
    def _is_float_token(token, float_symbols):
        if not token:
            return False
        token = token.strip()
        if token in float_symbols:
            return True
        if re.match(r'^\d+\.\d+(?:[eE][-+]?\d+)?f?$', token):
            return True
        if re.match(r'^\d+f$', token):
            return True
        return False

    @staticmethod
    def _count_fp_binary_ops(kernel, float_symbols):
        fp_count = 0
        

        loop_pattern = re.compile(r'for\s*\(\s*\w+\s*=.*?\)')
        loop_count = len(loop_pattern.findall(kernel))
        iteration_multiplier = max(1, loop_count * 10) 
        
      
        assign_pattern = re.compile(r'\b([A-Za-z_]\w*)\s*\[[^\]]+\]\s*=\s*([^;]+)')
        for m in assign_pattern.finditer(kernel):
            target = m.group(1)
            if target in float_symbols:
                rhs = m.group(2)
                for symbol in float_symbols:
                    if symbol in rhs:
                      
                        fp_count += len(re.findall(r'(?<!\+)\+(?![+=])', rhs))
                        fp_count += len(re.findall(r'(?<!-)\-(?![-=])', rhs))
                        fp_count += len(re.findall(r'(?<!\*)\*(?![=*])', rhs))
                        fp_count += len(re.findall(r'/(?![=/])', rhs))
                        break
        
       
        expr_pattern = re.compile(
            r'([A-Za-z_]\w*|\d+\.\d+(?:[eE][-+]?\d+)?f?|\d+f?)\s*([+\-*/])\s*([A-Za-z_]\w*|\d+\.\d+(?:[eE][-+]?\d+)?f?|\d+f?)'
        )
        for m in expr_pattern.finditer(kernel):
            op = m.group(2)
            if op == '+' and m.group(0).count('++'):
                continue
            if op == '-' and m.group(0).count('--'):
                continue
            left = m.group(1)
            right = m.group(3)
            if RooflineAnalyzer._is_float_token(left, float_symbols) or RooflineAnalyzer._is_float_token(right, float_symbols):
                fp_count += 1
        
    
        return fp_count * iteration_multiplier

    @staticmethod
    def _count_int_index_ops(kernel, float_symbols):
       
        int_count = 0
        
       
        index_pattern = re.compile(r'\b\w+\s*\[[^\]]+\]')
        int_count += len(index_pattern.findall(kernel))
        
   
        expr_pattern = re.compile(
            r'([A-Za-z_]\w*|\d+)\s*([+\-*/])\s*([A-Za-z_]\w*|\d+)'
        )
        for m in expr_pattern.finditer(kernel):
            op = m.group(2)
            if op == '+' and m.group(0).count('++'):
                continue
            if op == '-' and m.group(0).count('--'):
                continue
            left = m.group(1)
            right = m.group(3)
         
            if not RooflineAnalyzer._is_float_token(left, float_symbols) and not RooflineAnalyzer._is_float_token(right, float_symbols):
                int_count += 1
        
       
        bitwise_pattern = re.compile(r'([A-Za-z_]\w*|\d+)\s*([&|~^])\s*([A-Za-z_]\w*|\d+)')
        int_count += len(bitwise_pattern.findall(kernel))
        
       
        shift_pattern = re.compile(r'([A-Za-z_]\w*|\d+)\s*(<<|>>)\s*([A-Za-z_]\w*|\d+)')
        int_count += len(shift_pattern.findall(kernel))
        
       
        mod_pattern = re.compile(r'([A-Za-z_]\w*|\d+)\s*%\s*([A-Za-z_]\w*|\d+)')
        int_count += len(mod_pattern.findall(kernel))
        
        return int_count

    @staticmethod
    def _count_arithmetic_breakdown(kernel):
      
        add_count = len(re.findall(r'(?<!\+)\+(?![+=])', kernel))
        sub_count = len(re.findall(r'(?<!-)\-(?![-=])', kernel))
        mul_count = len(re.findall(r'(?<!\*)\*(?![=*])', kernel))
        div_count = len(re.findall(r'/(?![=/])', kernel))
        
      
        add_count -= len(re.findall(r'\+\+', kernel))
        sub_count -= len(re.findall(r'--', kernel))
        
        return {
            "add": max(0, add_count),
            "sub": max(0, sub_count),
            "mul": max(0, mul_count),
            "div": max(0, div_count),
        }

    @staticmethod
    def _count_special_math(kernel):
        counts = {}
        for func in RooflineAnalyzer.FUNC_FLOP_WEIGHTS:
            counts[func] = len(re.findall(rf'\b{func}\s*\(', kernel))
        return counts

    @staticmethod
    def _estimate_memory_bytes(kernel, pointer_param_sizes):
        load_bytes = 0
        store_bytes = 0

        for name, info in pointer_param_sizes.items():
            size = int(info.get("size", 4))
            store_pattern = rf'\b{name}\s*\[[^\]]+\]\s*='
            stores = len(re.findall(store_pattern, kernel))
            accesses = len(re.findall(rf'\b{name}\s*\[[^\]]+\]', kernel))
            loads = max(0, accesses - stores)

            store_bytes += stores * size
            load_bytes += loads * size

        total = load_bytes + store_bytes
        if total <= 0:
          
            generic_accesses = len(re.findall(r'\b\w+\s*\[[^\]]+\]', kernel))
            total = max(4, generic_accesses * 4)
            load_bytes = total
            store_bytes = 0

        return {
            "loads_bytes": load_bytes,
            "stores_bytes": store_bytes,
            "total_bytes": total,
        }

    @staticmethod
    def analyze_static(code):
        """
        Statically analyzes a CUDA kernel code to estimate its Arithmetic Intensity (FLOPs/Byte).
        """
        sanitized = RooflineAnalyzer._sanitize_kernel_source(code)
        kernels = RooflineAnalyzer._extract_global_kernels(sanitized)
        if not kernels:
            return {"arithmetic_intensity": 1.0, "bottleneck": "Memory-Bound (Estimated)", "reason": "Default fallback due to lack of detectable global kernels."}

        total_flops = 0.0
        total_bytes = 0.0
        op_breakdown = {
            "add": 0,
            "sub": 0,
            "mul": 0,
            "div": 0,
            "special_math": 0,
            "fma": 0,
            "fp_ops_estimated": 0,
            "int_index_ops_estimated": 0,
        }
        special_breakdown = {name: 0 for name in RooflineAnalyzer.FUNC_FLOP_WEIGHTS}
        memory_breakdown = {
            "loads_bytes": 0,
            "stores_bytes": 0,
            "total_bytes": 0,
        }

        for kernel in kernels:
            body = kernel["body"]
            pointer_sizes = RooflineAnalyzer._parse_pointer_params(kernel["params"])
            float_symbols = RooflineAnalyzer._extract_float_symbols(kernel["params"], body)

            arithmetic = RooflineAnalyzer._count_arithmetic_breakdown(body)
            special = RooflineAnalyzer._count_special_math(body)
            mem = RooflineAnalyzer._estimate_memory_bytes(body, pointer_sizes)
            fp_binary_ops = RooflineAnalyzer._count_fp_binary_ops(body, float_symbols)
            int_index_ops = RooflineAnalyzer._count_int_index_ops(body, float_symbols)

            op_breakdown["add"] += arithmetic["add"]
            op_breakdown["sub"] += arithmetic["sub"]
            op_breakdown["mul"] += arithmetic["mul"]
            op_breakdown["div"] += arithmetic["div"]
            op_breakdown["fma"] += special.get("fma", 0) + special.get("fmaf", 0)

            for name, count in special.items():
                special_breakdown[name] += count
                if name not in ("fma", "fmaf"):
                    op_breakdown["special_math"] += count

            memory_breakdown["loads_bytes"] += mem["loads_bytes"]
            memory_breakdown["stores_bytes"] += mem["stores_bytes"]
            memory_breakdown["total_bytes"] += mem["total_bytes"]

            core_ops = arithmetic["add"] + arithmetic["sub"] + arithmetic["mul"] + arithmetic["div"]
            special_flops = sum(special[name] * RooflineAnalyzer.FUNC_FLOP_WEIGHTS[name] for name in special)
            flops = max(1.0, core_ops + special_flops)
            fp_ops_kernel = max(0, fp_binary_ops + special.get("fma", 0) * 2 + special.get("fmaf", 0) * 2)
            int_index_ops_kernel = max(0, int_index_ops)
            op_breakdown["fp_ops_estimated"] += fp_ops_kernel
            op_breakdown["int_index_ops_estimated"] += int_index_ops_kernel
            total_flops += flops
            total_bytes += max(1.0, mem["total_bytes"])

        ai = total_flops / total_bytes
        bottleneck = "Memory-Bound" if ai < 2.0 else "Compute-Bound"
        memory_share = 1.0 / (1.0 + ai)
        compute_share = 1.0 - memory_share
        confidence = "medium" if memory_breakdown["total_bytes"] > 0 and total_flops > 1 else "low"
        
      
        total_estimated_ops = op_breakdown["fp_ops_estimated"] + op_breakdown["int_index_ops_estimated"]
        fp_int_ratio = op_breakdown["fp_ops_estimated"] / op_breakdown["int_index_ops_estimated"] if op_breakdown["int_index_ops_estimated"] > 0 else float('inf')
        
        return {
            "arithmetic_intensity": ai,
            "bottleneck": bottleneck,
            "estimated_flops": total_flops,
            "estimated_int_ops": op_breakdown["int_index_ops_estimated"],
            "estimated_total_ops": total_estimated_ops,
            "estimated_bytes": total_bytes,
            "fp_int_ratio": fp_int_ratio,
            "op_breakdown": op_breakdown,
            "special_function_breakdown": special_breakdown,
            "memory_breakdown": memory_breakdown,
            "model_shares": {
                "memory_bound_share": memory_share,
                "compute_bound_share": compute_share,
            },
            "formula": "Arithmetic Intensity = Estimated FLOPs / Estimated Memory Bytes",
            "analysis_confidence": confidence,
            "reason": (
                f"Static model: FLOPs={total_flops:.1f}, INT ops={op_breakdown['int_index_ops_estimated']:.1f}, "
                f"Bytes={total_bytes:.1f}, AI={ai:.4f}, FP/INT ratio={fp_int_ratio:.2f} "
                f"using weighted math op counts and pointer-based memory estimates."
            )
        }

    @staticmethod
    def analyze_ncu_csv(csv_path):
        """
        Dynamically analyzes NCU CSV output to calculate Roofline metrics.
        Separates floating-point operations from integer/index operations.
        """
        csv_path = os.path.normpath(csv_path)
        if not os.path.exists(csv_path):
            return None
        
        fadd = 0
        fmul = 0
        ffma = 0
        iadd = 0
        imul = 0
        imad = 0
        bytes_transferred = 0

        try:
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
            for r in rows:
                if len(r) < 4:
                    continue
                row_str = ",".join(r).lower()
            
             
                if "fadd" in row_str and "fp" in row_str:
                    try: fadd += float(r[-1].replace(',', ''))
                    except ValueError: pass
                elif "fmul" in row_str and "fp" in row_str:
                    try: fmul += float(r[-1].replace(',', ''))
                    except ValueError: pass
                elif "ffma" in row_str:
                    try: ffma += float(r[-1].replace(',', ''))
                    except ValueError: pass
               
                elif "iadd" in row_str and ("int" in row_str or "integer" in row_str):
                    try: iadd += float(r[-1].replace(',', ''))
                    except ValueError: pass
                elif "imul" in row_str and ("int" in row_str or "integer" in row_str):
                    try: imul += float(r[-1].replace(',', ''))
                    except ValueError: pass
                elif "imad" in row_str:
                    try: imad += float(r[-1].replace(',', ''))
                    except ValueError: pass
             
                elif "dram__bytes" in row_str or "dram__throughput" in row_str:
                    try: bytes_transferred += float(r[-1].replace(',', ''))
                    except ValueError: pass

            flops = fadd + fmul + (ffma * 2)
            int_ops = iadd + imul + (imad * 2)
            total_ops = flops + int_ops
            
            if bytes_transferred == 0:
                return None

            ai = total_ops / bytes_transferred
            bottleneck = "Memory-Bound" if ai < 2.0 else "Compute-Bound"

            return {
                "arithmetic_intensity": ai,
                "bottleneck": bottleneck,
                "measured_flops": flops,
                "measured_int_ops": int_ops,
                "measured_total_ops": total_ops,
                "measured_bytes": bytes_transferred,
                "fp_int_ratio": flops / int_ops if int_ops > 0 else float('inf'),
                "op_breakdown": {
                    "fadd": fadd,
                    "fmul": fmul,
                    "ffma": ffma,
                    "iadd": iadd,
                    "imul": imul,
                    "imad": imad,
                },
                "reason": f"NCU measured {flops:.0f} FLOPs, {int_ops:.0f} INT ops / {bytes_transferred:.0f} Bytes."
            }
        except Exception:
            return None

    @staticmethod
    def blend_static_dynamic(static_metrics, dynamic_metrics, dynamic_weight=0.65):
        if not static_metrics:
            return dynamic_metrics
        if not dynamic_metrics:
            blended = dict(static_metrics)
            blended["source"] = "static"
            return blended

        s_ai = float(static_metrics.get("arithmetic_intensity", 0.0) or 0.0)
        d_ai = float(dynamic_metrics.get("arithmetic_intensity", 0.0) or 0.0)
        weight = max(0.0, min(float(dynamic_weight), 1.0))
        blended_ai = (s_ai * (1.0 - weight)) + (d_ai * weight)

      
        s_flops = float(static_metrics.get("estimated_flops", 0.0) or 0.0)
        s_int_ops = float(static_metrics.get("estimated_int_ops", 0.0) or 0.0)
        d_flops = float(dynamic_metrics.get("measured_flops", 0.0) or 0.0)
        d_int_ops = float(dynamic_metrics.get("measured_int_ops", 0.0) or 0.0)
        
        blended_flops = (s_flops * (1.0 - weight)) + (d_flops * weight)
        blended_int_ops = (s_int_ops * (1.0 - weight)) + (d_int_ops * weight)
        blended_total_ops = blended_flops + blended_int_ops
        blended_fp_int_ratio = blended_flops / blended_int_ops if blended_int_ops > 0 else float('inf')

      
        s_bytes = float(static_metrics.get("estimated_bytes", 0.0) or 0.0)
        d_bytes = float(dynamic_metrics.get("measured_bytes", 0.0) or 0.0)
        blended_bytes = (s_bytes * (1.0 - weight)) + (d_bytes * weight)

        blended = dict(static_metrics)
        blended.update({
            "arithmetic_intensity": blended_ai,
            "bottleneck": "Memory-Bound" if blended_ai < 2.0 else "Compute-Bound",
            "source": "blended",
            "blend_weight_dynamic": weight,
            "static_arithmetic_intensity": s_ai,
            "dynamic_arithmetic_intensity": d_ai,
            "blended_flops": blended_flops,
            "blended_int_ops": blended_int_ops,
            "blended_total_ops": blended_total_ops,
            "blended_bytes": blended_bytes,
            "blended_fp_int_ratio": blended_fp_int_ratio,
            "dynamic_reason": dynamic_metrics.get("reason", ""),
            "dynamic_measured_flops": d_flops,
            "dynamic_measured_int_ops": d_int_ops,
            "dynamic_measured_bytes": d_bytes,
            "dynamic_fp_int_ratio": dynamic_metrics.get("fp_int_ratio"),
            "reason": (
                f"Blended model: AI = (1-w)*AI_static + w*AI_ncu, "
                f"w={weight:.2f}, AI_static={s_ai:.4f}, AI_ncu={d_ai:.4f}, AI_blended={blended_ai:.4f}. "
                f"FP/INT: static={s_flops:.0f}/{s_int_ops:.0f} (ratio={s_flops/s_int_ops if s_int_ops > 0 else float('inf'):.2f}), "
                f"NCU={d_flops:.0f}/{d_int_ops:.0f} (ratio={d_flops/d_int_ops if d_int_ops > 0 else float('inf'):.2f}), "
                f"blended={blended_flops:.0f}/{blended_int_ops:.0f} (ratio={blended_fp_int_ratio:.2f}). "
                f"Memory: static={s_bytes:.0f}B, NCU={d_bytes:.0f}B, blended={blended_bytes:.0f}B."
            ),
        })
        return blended
