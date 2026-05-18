import click
import json
import os
import difflib
from datetime import datetime
from .discover import check_environment
from .sandbox import CUDASandbox
from .llm_client import LLMClient

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.json')

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {"llm_url": "http://localhost:8080/completion"}

def save_report(report_data, filepath):
    with open(filepath, 'w') as f:
        json.dump(report_data, f, indent=2)

def print_diff(old_code, new_code):
    diff = difflib.unified_diff(old_code.splitlines(), new_code.splitlines(), lineterm='')
    for line in diff:
        if line.startswith('+') and not line.startswith('+++'):
            click.secho(line, fg='green')
        elif line.startswith('-') and not line.startswith('---'):
            click.secho(line, fg='red')
        elif line.startswith('^'):
            click.secho(line, fg='blue')
        else:
            click.echo(line)

@click.group()
def main():
    pass

@main.command()
def init():
    env_status = check_environment()
    click.echo(json.dumps(env_status, indent=2))
    config = load_config()
    click.echo(f"Config: {config}")

@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('-o', '--output', default=None)
@click.option('-i', '--iters', default=3)
@click.option('--target', default='latency')
@click.option('--retries', default=3)
@click.option('--fast-math', is_flag=True)
@click.option('-O', '--opt-level', default='3')
@click.option('--report', is_flag=True)
def optimize(input_file, output, iters, target, retries, fast_math, opt_level, report):
    config = load_config()
    if not output:
        output = f"optimized_{os.path.basename(input_file)}"
    
    with open(input_file, 'r') as f:
        original_code = f.read()

    flags = [f"-O{opt_level}"]
    if fast_math:
        flags.append("-use_fast_math")

    sandbox = CUDASandbox(input_file, flags=flags)
    llm = LLMClient(config['llm_url'])
    env_info = check_environment()
    
    best_time = float('inf')
    best_code = original_code
    current_code = original_code
    
    history = []

    for i in range(iters):
        click.secho(f"\n--- Iteration {i+1}/{iters} ---", fg="cyan", bold=True)
        prompt = llm.create_optimization_prompt(current_code, env_info, target, best_time, flags)
        new_code, gen_time = llm.generate_code(prompt)
        
        if not new_code:
            click.secho("Generation failed.", fg="red")
            continue

        temp_file = "temp_kernel.cu"
        with open(temp_file, 'w') as f:
            f.write(new_code)
        
        sandbox.file_path = temp_file
        compile_res = sandbox.compile()
        
        heal_attempts = 0
        while not compile_res['success'] and heal_attempts < retries:
            click.secho(f"Compile failed. Healing {heal_attempts+1}/{retries}...", fg="yellow")
            heal_prompt = llm.create_healing_prompt(new_code, compile_res['error_log'])
            new_code, gen_time = llm.generate_code(heal_prompt)
            if new_code:
                with open(temp_file, 'w') as f:
                    f.write(new_code)
                compile_res = sandbox.compile()
            heal_attempts += 1
            
        if not compile_res['success']:
            click.secho("Failed to heal code.", fg="red")
            continue
            
        prof_res = sandbox.profile_latency()
        latency = prof_res["latency"]
        click.echo(f"Latency: {latency} ms (Gen time: {gen_time:.2f}s)")
        
        history.append({
            "iteration": i+1,
            "latency": latency,
            "compile_success": True,
            "gen_time": gen_time
        })
        
        if latency < best_time:
            best_time = latency
            click.secho("New best latency!", fg="green", bold=True)
            print_diff(best_code, new_code)
            best_code = new_code
            
        current_code = new_code
        
    with open(output, 'w') as f:
        f.write(best_code)
        
    click.secho(f"\nOptimization complete. Best time: {best_time} ms.", fg="green", bold=True)
    click.echo(f"Total tokens predicted: {llm.total_tokens}")
    
    if report:
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "environment": env_info,
            "flags": flags,
            "best_latency": best_time,
            "history": history
        }
        report_file = f"report_{os.path.basename(input_file)}.json"
        save_report(report_data, report_file)
        click.echo(f"Report saved to {report_file}")

@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--markdown', is_flag=True)
def audit(input_file, markdown):
    config = load_config()
    with open(input_file, 'r') as f:
        code = f.read()
    llm = LLMClient(config['llm_url'])
    prompt = llm.create_audit_prompt(code)
    click.echo("Auditing...")
    analysis, _ = llm.generate_code(prompt)
    
    if markdown:
        out_name = f"audit_{os.path.basename(input_file)}.md"
        with open(out_name, 'w') as f:
            f.write(analysis)
        click.echo(f"Audit saved to {out_name}")
    else:
        click.echo(analysis)

if __name__ == '__main__':
    main()
