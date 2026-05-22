import re
import os
import csv

class RooflineAnalyzer:
    @staticmethod
    def analyze_static(code):
        """
        Statically analyzes a CUDA kernel code to estimate its Arithmetic Intensity (FLOPs/Byte).
        """
      
        kernels = re.findall(r'__global__\s+void\s+\w+\([^)]*\)\s*\{([\s\S]*?)\}', code)
        if not kernels:
            return {"arithmetic_intensity": 1.0, "bottleneck": "Memory-Bound (Estimated)", "reason": "Default fallback due to lack of detectable global kernels."}

        total_flops = 0
        total_bytes = 0

        for kernel in kernels:
          
            math_ops = len(re.findall(r'[\+\-\*\/]', kernel))
            math_funcs = len(re.findall(r'\b(sin|cos|exp|sqrt|pow|log|fma|fmaf)\b', kernel))
            flops = math_ops + (math_funcs * 2) 
            if flops == 0:
                flops = 1 

           

            mem_accesses = len(re.findall(r'\b\w+\[[^\]]+\]', kernel))
            mem_ptrs = len(re.findall(r'\*\w+', kernel))
          
            bytes_transferred = max(1, (mem_accesses + mem_ptrs) * 4)

            total_flops += flops
            total_bytes += bytes_transferred

        ai = total_flops / total_bytes
        bottleneck = "Memory-Bound" if ai < 2.0 else "Compute-Bound"
        
        return {
            "arithmetic_intensity": ai,
            "bottleneck": bottleneck,
            "estimated_flops": total_flops,
            "estimated_bytes": total_bytes,
            "reason": f"Static analysis estimated {total_flops} FLOPS vs {total_bytes} Bytes transferred."
        }

    @staticmethod
    def analyze_ncu_csv(csv_path):
        """
        Dynamically analyzes NCU CSV output to calculate Roofline metrics.
        """
        if not os.path.exists(csv_path):
            return None
        
        fadd = 0
        fmul = 0
        ffma = 0
        bytes_transferred = 0

        try:
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
            for r in rows:
                if len(r) < 4:
                    continue
                row_str = ",".join(r).lower()
            
                if "fadd" in row_str:
                    try: fadd += float(r[-1].replace(',', ''))
                    except ValueError: pass
                elif "fmul" in row_str:
                    try: fmul += float(r[-1].replace(',', ''))
                    except ValueError: pass
                elif "ffma" in row_str:
                    try: ffma += float(r[-1].replace(',', ''))
                    except ValueError: pass
                elif "dram__bytes" in row_str or "dram__throughput" in row_str:
                    try: bytes_transferred += float(r[-1].replace(',', ''))
                    except ValueError: pass

            flops = fadd + fmul + (ffma * 2)
            if bytes_transferred == 0:
                return None

            ai = flops / bytes_transferred
            bottleneck = "Memory-Bound" if ai < 2.0 else "Compute-Bound"

            return {
                "arithmetic_intensity": ai,
                "bottleneck": bottleneck,
                "measured_flops": flops,
                "measured_bytes": bytes_transferred,
                "reason": f"NCU measured {flops:.0f} FLOPS / {bytes_transferred:.0f} Bytes."
            }
        except Exception:
            return None
