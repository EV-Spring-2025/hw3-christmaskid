# This script is generated with the help of Github Copilot, using Claude Sonnet 4.

import os
import subprocess
import itertools
import time
from datetime import datetime


def run_simulation(model_path, config, material, output_base, 
                  n_grid=None, substep_dt=None, grid_v_damping_scale=None, softening=None):
    """Run a single simulation with specified parameters"""
    
    # Create experiment name based on parameters
    exp_name = f"{material}"
    if n_grid is not None:
        exp_name += f"_ngrid{n_grid}"
    if substep_dt is not None:
        exp_name += f"_dt{substep_dt}"
    if grid_v_damping_scale is not None:
        exp_name += f"_damping{grid_v_damping_scale}"
    if softening is not None:
        exp_name += f"_soft{softening}"
    
    output_path = os.path.join(output_base, exp_name)
    
    # Build command
    cmd = [
        "python", "gs_simulation.py",
        "--model_path", model_path,
        "--output_path", output_path,
        "--config", config,
        "--material", material,
        "--render_img",
        "--compile_video",
        "--white_bg"
    ]
    
    # Add optional parameters
    if n_grid is not None:
        cmd.extend(["--n_grid", str(n_grid)])
    if substep_dt is not None:
        cmd.extend(["--substep_dt", str(substep_dt)])
    if grid_v_damping_scale is not None:
        cmd.extend(["--grid_v_damping_scale", str(grid_v_damping_scale)])
    if softening is not None:
        cmd.extend(["--softening", str(softening)])
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {exp_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output path: {output_path}")
    print(f"{'='*60}", flush=True)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Setup combined log file for stdout and stderr
    combined_log = os.path.join(output_path, "simulation.log")
    
    # Run simulation
    start_time = time.time()
    try:
        with open(combined_log, 'w') as log_file:
            result = subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✓ Experiment completed successfully in {duration:.2f} seconds")
        
        # Save experiment log
        log_file = os.path.join(output_path, "experiment_log.txt")
        with open(log_file, 'w') as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Start time: {datetime.fromtimestamp(start_time)}\n")
            f.write(f"Duration: {duration:.2f} seconds\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Material: {material}\n")
            if n_grid is not None:
                f.write(f"n_grid: {n_grid}\n")
            if substep_dt is not None:
                f.write(f"substep_dt: {substep_dt}\n")
            if grid_v_damping_scale is not None:
                f.write(f"grid_v_damping_scale: {grid_v_damping_scale}\n")
            if softening is not None:
                f.write(f"softening: {softening}\n")
            f.write(f"\nOutput redirected to:\n")
            f.write(f"  combined log: {combined_log}\n")
        
        return True, duration
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✗ Experiment failed after {duration:.2f} seconds")
        print(f"Error: {e}")
        
        # Save error log
        error_log = os.path.join(output_path, "error_log.txt")
        with open(error_log, 'w') as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Start time: {datetime.fromtimestamp(start_time)}\n")
            f.write(f"Duration: {duration:.2f} seconds\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Error: {e}\n")
            f.write(f"\nOutput logs:\n")
            f.write(f"  combined log: {combined_log}\n")
        
        return False, duration

def run_evaluation(output_base, material, exp_name):
    """Run evaluation for a single material"""
    cmd = [
        "python", "psnr.py", 
        "--predicted_dir", os.path.join(output_base, exp_name),
        "--gt_output_dir", os.path.join(output_base, material),
    ]
    print(f"\nRunning evaluation for {exp_name}...")
    print(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Evaluation completed successfully:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed for {exp_name}:\n{e.stderr}")
        return False


def run_ablation_study():
    """Run complete ablation study"""
    
    # Configuration
    model_path = "./model/ficus_whitebg-trained/"
    config = "./config/ficus_config.json"
    output_base = "./ablation_results"
    
    # Create base output directory
    os.makedirs(output_base, exist_ok=True)
    
    # Parameter values to test
    all_materials = ["jelly", "sand", "snow", "metal", "foam", "plasticine"]
    materials = ["jelly", "sand"]
    n_grid_values = [25, 100, None]  # None = use default
    substep_dt_values = [5e-5, 2e-4, None]  # None = use default
    grid_v_damping_scale_values = [0.9995, 1.005, None]  # None = use default
    softening_values = [0.05, 0.2, None]  # None = use default
    
    # Track all experiments
    all_experiments = []
    successful_experiments = []
    failed_experiments = []
    total_duration = 0
    
    print("Starting PhysGaussian Ablation Study")
    print(f"Model path: {model_path}")
    print(f"Config: {config}")
    print(f"Output base: {output_base}")
    print(f"Materials: {materials}")
    
    study_start_time = time.time()
    
    # Run baseline experiments (all defaults)
    for material in all_materials:
        exp_name = f"{material}"
        all_experiments.append(exp_name)
        
        success, duration = run_simulation(
            model_path, config, material, output_base
        )
        
        total_duration += duration
        if success:
            successful_experiments.append(exp_name)
        else:
            failed_experiments.append(exp_name)
    
    # Run n_grid ablation
    for material in materials:
        for n_grid in n_grid_values:
            if n_grid is not None:  # Skip default (already tested in baseline)
                exp_name = f"{material}_ngrid{n_grid}"
                all_experiments.append(exp_name)
                
                success, duration = run_simulation(
                    model_path, config, material, output_base, n_grid=n_grid
                )
                
                total_duration += duration
                if success:
                    successful_experiments.append(exp_name)
                else:
                    failed_experiments.append(exp_name)

                eval_success = run_evaluation(output_base, material, exp_name)
                if eval_success:
                    print(f"Evaluation for {exp_name} completed successfully.")
                else:
                    print(f"Evaluation for {exp_name} failed.")
    
    # Run substep_dt ablation
    for material in materials:
        for substep_dt in substep_dt_values:
            if substep_dt is not None:  # Skip default (already tested in baseline)
                exp_name = f"{material}_dt{substep_dt}"
                all_experiments.append(exp_name)
                
                success, duration = run_simulation(
                    model_path, config, material, output_base, substep_dt=substep_dt
                )
                
                total_duration += duration
                if success:
                    successful_experiments.append(exp_name)
                else:
                    failed_experiments.append(exp_name)
    
                eval_success = run_evaluation(output_base, material, exp_name)
                if eval_success:
                    print(f"Evaluation for {exp_name} completed successfully.")
                else:
                    print(f"Evaluation for {exp_name} failed.")

    # Run grid_v_damping_scale ablation
    for material in materials:
        for damping in grid_v_damping_scale_values:
            if damping is not None:  # Skip default (already tested in baseline)
                exp_name = f"{material}_damping{damping}"
                all_experiments.append(exp_name)
                
                success, duration = run_simulation(
                    model_path, config, material, output_base, 
                    grid_v_damping_scale=damping
                )
                
                total_duration += duration
                if success:
                    successful_experiments.append(exp_name)
                else:
                    failed_experiments.append(exp_name)
    
                eval_success = run_evaluation(output_base, material, exp_name)
                if eval_success:
                    print(f"Evaluation for {exp_name} completed successfully.")
                else:
                    print(f"Evaluation for {exp_name} failed.")
                    
    # Run softening ablation
    for material in materials:
        for soft in softening_values:
            if soft is not None:  # Skip default (already tested in baseline)
                exp_name = f"{material}_soft{soft}"
                all_experiments.append(exp_name)
                
                success, duration = run_simulation(
                    model_path, config, material, output_base, softening=soft
                )
                
                total_duration += duration
                if success:
                    successful_experiments.append(exp_name)
                else:
                    failed_experiments.append(exp_name)
    
                eval_success = run_evaluation(output_base, material, exp_name)
                if eval_success:
                    print(f"Evaluation for {exp_name} completed successfully.")
                else:
                    print(f"Evaluation for {exp_name} failed.")
                    

    study_end_time = time.time()
    study_duration = study_end_time - study_start_time
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*80}")
    print(f"Total experiments: {len(all_experiments)}")
    print(f"Successful: {len(successful_experiments)}")
    print(f"Failed: {len(failed_experiments)}")
    print(f"Total duration: {study_duration:.2f} seconds ({study_duration/3600:.2f} hours)")
    print(f"Average per experiment: {study_duration/len(all_experiments):.2f} seconds")
    
    # Save summary report
    summary_file = os.path.join(output_base, "ablation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("PhysGaussian Ablation Study Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Study completed: {datetime.fromtimestamp(study_end_time)}\n")
        f.write(f"Total duration: {study_duration:.2f} seconds ({study_duration/3600:.2f} hours)\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Config: {config}\n\n")
        
        f.write(f"Total experiments: {len(all_experiments)}\n")
        f.write(f"Successful: {len(successful_experiments)}\n")
        f.write(f"Failed: {len(failed_experiments)}\n\n")
        
        f.write("Successful experiments:\n")
        for exp in successful_experiments:
            f.write(f"  ✓ {exp}\n")
        
        if failed_experiments:
            f.write("\nFailed experiments:\n")
            for exp in failed_experiments:
                f.write(f"  ✗ {exp}\n")
        
        f.write("\nParameter combinations tested:\n")
        f.write(f"  Materials: {materials}\n")
        f.write(f"  n_grid: {[x for x in n_grid_values if x is not None]} + default\n")
        f.write(f"  substep_dt: {[x for x in substep_dt_values if x is not None]} + default\n")
        f.write(f"  grid_v_damping_scale: {[x for x in grid_v_damping_scale_values if x is not None]} + default\n")
        f.write(f"  softening: {[x for x in softening_values if x is not None]} + default\n")
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"Individual experiment results in: {output_base}")


if __name__ == "__main__":
    run_ablation_study()
