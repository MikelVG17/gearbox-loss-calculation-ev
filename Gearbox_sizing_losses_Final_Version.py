import json
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# Define standard modules
first_choice_modules = modules = [1, 1.125, 1.25, 1.375, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 25, 28]


@dataclass
class GearStage:
    """
    Represents a single stage in a gearbox, including details on gear type, dimensions, and operational parameters.
    """
    type: str = None  # Indicates whether the stage is a 'shift' or 'ratio'
    ratio: float = None  # Default to None to handle shift stages properly
    ratios: List[float] = field(default_factory=list)
    realisation: str = None  # Represents the 'realisation' field from JSON (e.g., 'PGS', 'SG')
    operation: str = field(default=None)
    driving_diameter: float = field(default=None)
    driven_diameter: float = field(default=None)
    driving_teeth: int = field(default=None)
    driven_teeth: int = field(default=None)
    sun_diameter: float = field(default=None)
    planet_diameter: float = field(default=None)
    ring_diameter: float = field(default=None)
    sun_teeth: int = field(default=None)
    planet_teeth: int = field(default=None)
    ring_teeth: int = field(default=None)
    module: float = field(default=None)
    real_operation_ratio: float = field(default=None)
    num_planets: int = field(default=3)
    gear_type: str = None  # Only used for SG gears

def load_json(file_path: str) -> dict:
    """
    Load a JSON file and return its content as a dictionary.

    Parameters:
    file_path (str): The path to the JSON file to be loaded.

    Returns:
    dict: The contents of the loaded JSON file.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File {file_path} is not a valid JSON file.")
        sys.exit(1)

def validate_gear_ratios(gear_stages):
    """
    Validate that gear ratios for planetary gear stages are within valid bounds.

    Parameters:
    gear_stages (List[GearStage]): A list of GearStage objects to validate.

    Returns:
    None
    """
    for stage in gear_stages:
        if stage.realisation == 'PGS':
            # Check both 'ratio' and 'ratios'
            if stage.ratio is not None:
                if not (1.5 <= stage.ratio <= 4):
                    print(f"Error: The ratio {stage.ratio} for the planetary gear stage is invalid. Ratios for PGS should be between 1.5 and 4.")
                    sys.exit(1)
            elif stage.ratios:
                for r in stage.ratios:
                    if not (1.5 <= r <= 4):
                        print(f"Error: The ratio {r} for the planetary gear stage is invalid. Ratios for PGS should be between 1.5 and 4.")
                        sys.exit(1)

def extract_configuration(data: dict) -> dict:
    """
    Extract the configuration dictionary to get necessary parameters.

    Parameters:
    data (dict): The input data containing configuration settings.

    Returns:
    dict: The extracted and validated configuration dictionary.
    """
    try:
        material_strength = data["material_strength"]
        security_factor = data["security_factor"]
        configurations = data["configurations"]
    except KeyError as e:
        print(f"Error: Missing key {e} in the configuration file.")
        sys.exit(1)

    print("Choose a configuration:")
    for i, config in enumerate(configurations, start=1):
        print(f"{i}: {config}")

    try:
        choice = int(input("Enter the number of the desired configuration: "))
        chosen_config_key = list(configurations.keys())[choice - 1]
    except (ValueError, IndexError):
        print("Error: Invalid choice.")
        sys.exit(1)

    configuration = configurations[chosen_config_key]

    if 'gear_stages' not in configuration:
        print(f"Error: 'gear_stages' key is missing in the chosen configuration.")
        sys.exit(1)

    gear_stages = []
    for stage in configuration['gear_stages']:
        gear_stage = GearStage(
            type=stage['type'],  # Assign the 'type' from JSON (e.g., 'shift' or 'ratio')
            ratio=stage.get('ratio', None),  # Handle a single ratio if present
            ratios=stage.get('ratios', []),  # Handle multiple ratios if it's a shift stage
            realisation=stage['realisation'],  # Assign the 'realisation' from JSON (e.g., 'PGS', 'SG')
            operation=stage.get('operation', ""),
            num_planets=stage.get('num_planets', 3),
            gear_type=stage.get('gear_type', "helical") if stage['realisation'] == 'SG' else None  # Set gear_type only for SG
        )
        gear_stages.append(gear_stage)

    # Validate the ratios for planetary gear stages
    validate_gear_ratios(gear_stages)

    configuration.update({
        'gear_stages': gear_stages,
        'material_strength': material_strength,
        'security_factor': security_factor,
        'N_open_friction_elements': configuration.get("N_open_friction_elements", 0),
        'N_open_dog_clutches': configuration.get("N_open_dog_clutches", 0),
        'N_open_synchronisation_gaps': configuration.get("N_open_synchronisation_gaps", 0),
        'N_differential_cages': configuration.get("N_differential_cages", 1),
        'N_planetary_carriers': configuration.get("N_planetary_carriers", 0),
        'N_bearings': configuration.get("N_bearings", 0),
        'N_oil_pumps': configuration.get("N_oil_pumps", 0),
        'speeds': configuration.get("speeds", 1),
        'number_of_stages': configuration.get("number_of_stages", 1),
        'RPM_range': configuration.get("RPM_range", 1000),
        'output_torque': configuration.get("output_torque", 100)
    })

    return configuration

def load_configuration(file_path: str) -> dict:
    """
    Load and extract gearbox configuration from a JSON file.

    Parameters:
    file_path (str): The path to the configuration JSON file.

    Returns:
    dict: The extracted configuration dictionary.
    """
    data = load_json(file_path)
    return extract_configuration(data)


def find_nearest_higher_module_index(m: float, modules: List[float]) -> int:
    """
    Find the index of the nearest higher module in the standard module list.

    Parameters:
    m (float): The module value to find a match for.
    modules (List[float]): A list of standard modules to search.

    Returns:
    int: The index of the nearest higher module.
    """
    for i, module in enumerate(modules):
        if module >= m:
            return i
    return len(modules) - 1

def size_multistage_gearbox(output_torque: float, gear_stages: List[GearStage], material_strength: float, security_factor: float, smallest_shift_ratio: float) -> List[Dict[str, float]]:
    """
    Determine the parameters for each stage of a multistage gearbox.

    Parameters:
    output_torque (float): The output torque for the gearbox.
    gear_stages (List[GearStage]): A list of GearStage objects representing the stages.
    material_strength (float): The material strength used in stress calculations.
    security_factor (float): The safety factor applied in stress calculations.
    smallest_shift_ratio (float): The smallest shift ratio in the gearbox.

    Returns:
    List[Dict[str, float]]: A list of dictionaries representing the calculated parameters for each stage.
    """
    stages = len(gear_stages)
    T_current = output_torque
    results = []

    for stage in reversed(range(stages)):
        gear_stage = gear_stages[stage]

        # Ensure the ratio is properly set
        if gear_stage.type == "shift":  # Check if it's a shift stage
            if not gear_stage.ratios:
                raise ValueError(f"Shift stage '{gear_stage}' is missing 'ratios'. Please provide valid ratios in the configuration.")
        elif gear_stage.ratio is None:
            raise ValueError(f"Ratio for stage of type '{gear_stage.type}' is not set. Please check the configuration.")

        result = size_gearbox_stage(T_current, gear_stage, material_strength, security_factor, stage + 1)
        results.append(result)
        
        # Update T_current for the next stage
        if gear_stage.realisation == 'SG' or (gear_stage.realisation == 'PGS' and gear_stage.operation in ['12']):
            T_current = T_current / gear_stage.ratio
        elif gear_stage.realisation == 'PGS':
            T_current = T_current / gear_stage.real_operation_ratio
        
        # Check if this stage is a ratio stage and update the max sizes accordingly
        if gear_stage.ratio == smallest_shift_ratio or gear_stage.real_operation_ratio == smallest_shift_ratio:
            update_max_sizes(gear_stage)

    return results

def initialize_stress_analysis(module_index, F_t, pitch_d_initial, beta, material_strength, security_factor):
    """
    Initialize stress parameters for a gear stage.

    Parameters:
    module_index (int): The index of the module being analyzed.
    F_t (float): The tangential force applied to the gear.
    pitch_d_initial (float): The initial pitch diameter of the gear.
    beta (float): The helix angle in degrees.
    material_strength (float): The strength of the material.
    security_factor (float): The safety factor applied to the stress calculations.

    Returns:
    tuple: A tuple containing the face width (b), calculated stress (sigma), allowable stress, and the Lewis Form Factor (Y).
    """
    
    # Constants
    Y = 0.35  # Lewis Form Factor, typically based on gear tooth profile
    b = round(((10 * first_choice_modules[module_index]) /  math.cos(math.radians(beta))), 1)  # Face width
    
    # Stress Calculation (sigma)
    sigma = abs((F_t * 1000) / (b * first_choice_modules[module_index] * Y))  # Stress in MPa
    allowable_stress = material_strength / security_factor
    
    # Log initial values for debugging purposes
    print(f"Initial Sigma: {sigma}, Allowable Stress: {allowable_stress}, Initial Module: {first_choice_modules[module_index]}")
    
    return b, sigma, allowable_stress, Y


def size_gearbox_stage(T, gear_stage, material_strength, security_factor, stage: int) -> dict:
    """
    Determine parameters for a single gearbox stage and track sigma and module.

    Parameters:
    T (float): Torque for the stage.
    gear_stage (GearStage): The GearStage object representing the current stage.
    material_strength (float): The material strength used in stress calculations.
    security_factor (float): The safety factor applied to the stage.
    stage (int): The stage number in the gearbox.

    Returns:
    dict: A dictionary containing the calculated parameters for the stage.
    """
    # Initialize parameters and constants
    ratio = gear_stage.ratio
    realisation = gear_stage.realisation
    operation = gear_stage.operation
    num_planets = gear_stage.num_planets
    pitch_d_initial = 100
    z_initial = 25
    m_initial = 0.0
    beta = 20
    Z_1, Z_2 = 0, 0  # Ensure Z_1 and Z_2 are initialized
    highest_torque_gear = None  # Initialize to None for non-planetary gears (SG)

    if realisation == 'SG':
        # Calculations for simple gear
        m_initial = pitch_d_initial * math.cos(math.radians(beta)) / z_initial
        z_driving = z_initial
        z_driven = round(z_driving * ratio)
        d_driving = pitch_d_initial
        d_driven = d_driving * ratio
        gear_stage.driving_diameter = d_driving
        gear_stage.driven_diameter = d_driven
        gear_stage.driving_teeth = z_driving
        gear_stage.driven_teeth = z_driven
        F_t = round(T * 2 / (pitch_d_initial * math.cos(math.radians(beta))), 2)
    elif realisation == 'PGS':
        # Calculations for planetary gear
        Z_1 = z_initial
        Z_2 = round(ratio * Z_1)
        ratio, T_sun, T_ring, T_carrier = calculate_pgs_torque_ratios(T, operation, Z_1, Z_2)
        gear_stage.real_operation_ratio = ratio
        m_initial, F_t, highest_torque_gear = determine_highest_torque_gear(T_sun, T_ring, T_carrier, Z_1, Z_2, pitch_d_initial, beta, num_planets)
        T = max(T_sun, T_ring, T_carrier)
    else:
        raise ValueError("Unsupported gear type")

    # Find the module index and start stress analysis
    module_index = find_nearest_higher_module_index(m_initial, first_choice_modules)
    m_initial = first_choice_modules[module_index]
    b, sigma, allowable_stress, Y = initialize_stress_analysis(module_index, F_t, pitch_d_initial, beta, material_strength, security_factor)

    # Refinement loop to find the suitable module
    while sigma > allowable_stress and module_index < len(first_choice_modules) - 1:       
        module_index, m_initial, b, z_initial, F_t = refine_stress_parameters(
            module_index, m_initial, z_initial, ratio, realisation, highest_torque_gear, T, num_planets, beta
        )
        
        sigma = abs((F_t * 1000) / (b * m_initial * Y)) # Recalculate sigma

    if sigma > allowable_stress:
        raise ValueError("No suitable module found that meets the stress condition.")

    # Apply final gear dimensions
    gear_stage.module = m_initial
    apply_gear_dimensions(gear_stage, m_initial, realisation, z_initial, Z_1, Z_2, ratio, beta, num_planets)

    print(f"Stage {stage} | Final Module: {m_initial}, Final Sigma: {sigma}, Allowable Stress: {allowable_stress}")

    return compile_stage_results(stage, realisation, F_t, m_initial, pitch_d_initial, b, gear_stage)

def calculate_pgs_torque_ratios(T, operation, Z_1, Z_2):
    """
    Calculate the torque ratios for a planetary gear system.

    Parameters:
    T (float): Torque input for the system.
    operation (str): The type of planetary gear operation mode.
    Z_1 (int): Number of teeth in the sun gear.
    Z_2 (int): Number of teeth in the ring gear.

    Returns:
    tuple: A tuple containing the gear ratio, torque for the sun gear (T_sun), 
           torque for the ring gear (T_ring), and torque for the carrier (T_carrier).
    """
    if operation == '1s':
        # Sun gear is the input, carrier is the output
        ratio = 1 + (Z_2 / Z_1)
        T_carrier = T  # Carrier is the output
        T_sun = T_carrier / ratio  # Sun is the input, reverse the ratio to get input torque
        T_ring = 0  # Ring is stationary

    elif operation == 's1':
        # Carrier is the input, sun gear is the output
        ratio = 1 / (1 + (Z_2 / Z_1))
        T_sun = T  # Sun gear is the output
        T_carrier = T_sun / ratio  # Carrier is the input, reverse the ratio to get input torque
        T_ring = 0  # Ring is stationary

    elif operation == '2s':
        # Ring gear is the input, carrier is the output
        ratio = -(1 + (Z_1 / Z_2))
        T_carrier = T  # Carrier is the output
        T_ring = T_carrier / ratio  # Ring is the input
        T_sun = 0  # Sun is stationary

    elif operation == 's2':
        # Carrier is the input, ring gear is the output
        ratio = -1 / (1 + (Z_1 / Z_2))
        T_ring = T  # Ring is the output
        T_carrier = T_ring / ratio  # Carrier is the input
        T_sun = 0  # Sun is stationary

    elif operation == '12':
        # Sun gear is the input, ring gear is the output
        ratio = Z_2 / Z_1
        T_ring = T  # Ring is the output
        T_sun = T_ring / ratio  # Sun is the input
        T_carrier = 0  # Carrier is stationary

    else:
        raise ValueError("Unsupported operation mode")
    
    return ratio, T_sun, T_ring, T_carrier


def determine_highest_torque_gear(T_sun, T_ring, T_carrier, Z_1, Z_2, pitch_d_initial, beta, num_planets):
    """
    Determine the gear with the highest torque in a planetary gear system.

    Parameters:
    T_sun (float): Torque on the sun gear.
    T_ring (float): Torque on the ring gear.
    T_carrier (float): Torque on the carrier.
    Z_1 (int): Number of teeth on the sun gear.
    Z_2 (int): Number of teeth on the ring gear.
    pitch_d_initial (float): Initial pitch diameter for the gear.
    beta (float): Helix angle in degrees.
    num_planets (int): Number of planet gears.

    Returns:
    tuple: A tuple containing the module, tangential force (F_t), and the gear type with the highest torque.
    """
    max_torque = max(T_sun, T_ring, T_carrier)
    highest_torque_gear = None
    
    # Set a minimum number of teeth for the planet gears
    min_teeth = 8  # Minimum allowed number of teeth for planet gears
    
    if max_torque == T_sun:
        highest_torque_gear = "sun"
        m_initial = pitch_d_initial * math.cos(math.radians(beta)) / Z_1
        r_sun = pitch_d_initial / 2
        F_t = round(T_sun / (r_sun * math.cos(math.radians(beta))), 2)
    
    elif max_torque == T_ring:
        highest_torque_gear = "ring"
        # Apply a correction factor to make the Ring Gear larger than the Sun Gear
        correction_factor_ring = 1.2  # 20% correction factor for the Ring diameter
        pitch_d_initial_ring = pitch_d_initial * correction_factor_ring
        
        m_initial = pitch_d_initial_ring * math.cos(math.radians(beta)) / Z_2
        r_ring = pitch_d_initial_ring / 2
        F_t = round(T_ring / (r_ring * math.cos(math.radians(beta))), 2)
    
    elif max_torque == T_carrier:
        highest_torque_gear = "planet"
        Z_p = round((Z_2 - Z_1) / 2)
        
        # Ensure the planet gear teeth count doesn't go below the minimum threshold
        if Z_p < min_teeth:
            Z_p = min_teeth
        
        # Adjustment: reduce the initial pitch diameter for the planet
        reduction_factor = 0.2  # Reduce planet's initial pitch diameter
        pitch_d_initial_planet = pitch_d_initial * reduction_factor
        
        m_initial = pitch_d_initial_planet * math.cos(math.radians(beta)) / Z_p
        r_planet = pitch_d_initial_planet / 2
        F_t = round((T_carrier / num_planets) / (r_planet * math.cos(math.radians(beta))), 2)
    
    return m_initial, F_t, highest_torque_gear




def refine_stress_parameters(module_index, m_initial, z_initial, ratio, realisation, highest_torque_gear, T, num_planets, beta):
    """
    Update stress parameters during the iteration to find a suitable module size.

    Parameters:
    module_index (int): The index of the module being analyzed.
    m_initial (float): The initial module value.
    z_initial (int): The initial number of teeth on the gear.
    ratio (float): Gear ratio for the stage.
    realisation (str): The type of gear realization (SG or PGS).
    highest_torque_gear (str): The gear type with the highest torque.
    T (float): The torque being applied to the gear.
    num_planets (int): Number of planet gears in the system.
    beta (float): Helix angle in degrees.

    Returns:
    tuple: A tuple containing the updated module index, module size, face width, number of teeth, and tangential force.
    """
    
    # Move to the next higher module in the list
    module_index += 1
    if module_index >= len(first_choice_modules):
        raise ValueError("No suitable module found that meets the stress condition.")
    
    m_initial = first_choice_modules[module_index]
    b = round(((10 * m_initial) /  math.cos(math.radians(beta))), 1)  # Update face width
    
    # Increase the teeth count to maintain the ratio
    z_initial += 1

    if realisation == 'SG':  # Simple Gear Calculation
        z_driving = z_initial
        z_driven = round(z_driving * ratio)
        pitch_diameter = m_initial * z_driving / math.cos(math.radians(beta))
        pitch_radius = pitch_diameter / 2
        F_t = round(T / (pitch_radius * math.cos(math.radians(beta))), 2)

    elif highest_torque_gear == "sun":  # Refinement for Sun Gear in PGS
        Z_1 = z_initial
        Z_2 = round(Z_1 * ratio)
        pitch_diameter = m_initial * Z_1 / math.cos(math.radians(beta))
        pitch_radius = pitch_diameter / 2
        F_t = round(T / (pitch_radius * math.cos(math.radians(beta))), 2)

    elif highest_torque_gear == "ring":  # Refinement for Ring Gear in PGS
        Z_2 = round(Z_1 * ratio)
        Z_1 = round(Z_2 / ratio)
        Z_p = round((Z_2 - Z_1) / 2)
        pitch_diameter = m_initial * Z_2 / math.cos(math.radians(beta))
        pitch_radius = pitch_diameter / 2
        F_t = round(T / (pitch_radius * math.cos(math.radians(beta))), 2)

    elif highest_torque_gear == "planet":  # Refinement for Planet Gear in PGS
        Z_1 = z_initial
        Z_2 = round(Z_1 * ratio)
        Z_p = round((Z_2 - Z_1) / 2)
        # Ensure planet teeth remain above the minimum
        min_teeth = 8
        if Z_p < min_teeth:
            Z_p = min_teeth
        pitch_diameter = m_initial * Z_p / math.cos(math.radians(beta))
        pitch_radius = pitch_diameter / 2
        F_t = round((T / num_planets) / (pitch_radius * math.cos(math.radians(beta))), 2)

    return module_index, m_initial, b, z_initial, F_t


def apply_gear_dimensions(gear_stage, m_initial, realisation, z_initial, Z_1, Z_2, ratio, beta, num_planets):
    """
    Update the gear stage with calculated parameters.

    Parameters:
    gear_stage (GearStage): The GearStage object to be updated.
    m_initial (float): The initial module value.
    realisation (str): The type of gear realization (SG or PGS).
    z_initial (int): The initial number of teeth on the gear.
    Z_1 (int): The number of teeth on the sun gear.
    Z_2 (int): The number of teeth on the ring gear.
    ratio (float): The gear ratio for the stage.
    beta (float): Helix angle in degrees.
    num_planets (int): Number of planetary gears.

    Returns:
    None
    """
    min_teeth = 8  # Minimum allowed number of teeth for planet gears

    if realisation == 'SG':  # Simple Gear Calculation
        z_driving = z_initial
        z_driven = round(z_driving * ratio)
        gear_stage.driving_diameter = m_initial * z_driving / math.cos(math.radians(beta))
        gear_stage.driven_diameter = m_initial * z_driven / math.cos(math.radians(beta))
        gear_stage.driving_teeth = z_driving
        gear_stage.driven_teeth = z_driven
        gear_stage.module = m_initial
    
    elif realisation == 'PGS':  # Planetary Gear Calculation
        Z_1 = round(Z_1)
        Z_2 = round(Z_2)
        Z_p = round((Z_2 - Z_1) / 2)
        
        # Enforce minimum teeth constraint for planet gears
        if Z_p < min_teeth:
            scale_factor = min_teeth / Z_p
            Z_1 = round(Z_1 * scale_factor)
            Z_2 = round(Z_2 * scale_factor)
            Z_p = min_teeth
        
        gear_stage.sun_diameter = m_initial * Z_1 / math.cos(math.radians(beta))
        gear_stage.ring_diameter = m_initial * Z_2 / math.cos(math.radians(beta))
        gear_stage.planet_diameter = (gear_stage.ring_diameter - gear_stage.sun_diameter) / 2
        gear_stage.sun_teeth = Z_1
        gear_stage.ring_teeth = Z_2
        gear_stage.planet_teeth = Z_p
        gear_stage.module = m_initial


def compile_stage_results(stage, realisation, F_t, m_initial, pitch_d_initial, b, gear_stage):
    """
    Create a result dictionary for a gearbox stage.

    Parameters:
    stage (int): The stage number.
    realisation (str): The type of gear realization (SG or PGS).
    F_t (float): The tangential force applied to the gear.
    m_initial (float): The module size for the gear.
    pitch_d_initial (float): The initial pitch diameter for the gear.
    b (float): The face width of the gear.
    gear_stage (GearStage): The GearStage object for this stage.

    Returns:
    dict: A dictionary containing the results for the stage.
    """
    return {
        "stage": stage,
        "gear_type": realisation,
        "tangential_force": F_t,
        "module": m_initial,
        "initial_pitch_diameter": pitch_d_initial,
        "face_width": b,
        "driving_diameter": gear_stage.driving_diameter,
        "driven_diameter": gear_stage.driven_diameter,
        "driving_teeth": gear_stage.driving_teeth,
        "driven_teeth": gear_stage.driven_teeth,
        "sun_diameter": gear_stage.sun_diameter,
        "planet_diameter": gear_stage.planet_diameter,
        "ring_diameter": gear_stage.ring_diameter,
        "sun_teeth": gear_stage.sun_teeth,
        "planet_teeth": gear_stage.planet_teeth,
        "ring_teeth": gear_stage.ring_teeth
    }

def update_max_sizes(gear_stage: GearStage):
    """
    Update the sizes for the ratio stages based on the current gear stage.

    Parameters:
    gear_stage (GearStage): The GearStage object representing the current stage.

    Returns:
    None
    """
    global max_ratio_stage_sizes

    if gear_stage.module is not None and gear_stage.module > max_ratio_stage_sizes.get("module", 0):
        max_ratio_stage_sizes["module"] = gear_stage.module

    if gear_stage.driving_diameter is not None and gear_stage.driving_diameter > max_ratio_stage_sizes.get("driving_diameter", 0):
        max_ratio_stage_sizes["driving_diameter"] = gear_stage.driving_diameter

    if gear_stage.driven_diameter is not None and gear_stage.driven_diameter > max_ratio_stage_sizes.get("driven_diameter", 0):
        max_ratio_stage_sizes["driven_diameter"] = gear_stage.driven_diameter

    if gear_stage.driving_teeth is not None and gear_stage.driving_teeth > max_ratio_stage_sizes.get("driving_teeth", 0):
        max_ratio_stage_sizes["driving_teeth"] = gear_stage.driving_teeth

    if gear_stage.driven_teeth is not None and gear_stage.driven_teeth > max_ratio_stage_sizes.get("driven_teeth", 0):
        max_ratio_stage_sizes["driven_teeth"] = gear_stage.driven_teeth

    if gear_stage.sun_diameter is not None and gear_stage.sun_diameter > max_ratio_stage_sizes.get("sun_diameter", 0):
        max_ratio_stage_sizes["sun_diameter"] = gear_stage.sun_diameter

    if gear_stage.planet_diameter is not None and gear_stage.planet_diameter > max_ratio_stage_sizes.get("planet_diameter", 0):
        max_ratio_stage_sizes["planet_diameter"] = gear_stage.planet_diameter

    if gear_stage.ring_diameter is not None and gear_stage.ring_diameter > max_ratio_stage_sizes.get("ring_diameter", 0):
        max_ratio_stage_sizes["ring_diameter"] = gear_stage.ring_diameter

    if gear_stage.sun_teeth is not None and gear_stage.sun_teeth > max_ratio_stage_sizes.get("sun_teeth", 0):
        max_ratio_stage_sizes["sun_teeth"] = gear_stage.sun_teeth

    if gear_stage.planet_teeth is not None and gear_stage.planet_teeth > max_ratio_stage_sizes.get("planet_teeth", 0):
        max_ratio_stage_sizes["planet_teeth"] = gear_stage.planet_teeth

    if gear_stage.ring_teeth is not None and gear_stage.ring_teeth > max_ratio_stage_sizes.get("ring_teeth", 0):
        max_ratio_stage_sizes["ring_teeth"] = gear_stage.ring_teeth

drag_torque_values = {
    'Open friction element': 0.5,
    'Open dog clutch': 0.08,
    'Open synchronisation gap': 0.08,
    'Differential cage': 1.5,
    'Planetary carrier and bearings': 0.37,
    'Mechanical oil pump': 2.5
}

def load_independent_losses(values_input: dict, total_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate load-independent losses for the gearbox.

    Parameters:
    values_input (dict): A dictionary of input values needed for the calculation.
    total_ratio (float): The total gear ratio for the gearbox.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Two numpy arrays containing total torque losses and total power losses.
    """
    base_speeds = 7
    base_open_friction_elements = 1
    base_open_dog_clutches = 0
    base_open_synchronization_gaps = 6
    base_differentials = 1
    base_planetary_carriers = 0
    base_bearings = 0
    base_mechanical_oil_pumps = 0

    drag_open_friction_elements = (values_input['N_open_friction_elements'] - base_open_friction_elements) * drag_torque_values['Open friction element']
    drag_open_dog_clutches = (values_input['N_open_dog_clutches'] - base_open_dog_clutches) * drag_torque_values['Open dog clutch']
    drag_open_synchronisation_gaps = (values_input['N_open_synchronisation_gaps'] - base_open_synchronization_gaps) * drag_torque_values['Open synchronisation gap']
    drag_differential_cages = (values_input['N_differential_cages'] - base_differentials) * drag_torque_values['Differential cage']
    drag_planetary_carriers = (values_input['N_planetary_carriers'] - base_planetary_carriers) * drag_torque_values['Planetary carrier and bearings']
    drag_bearings = (values_input['N_bearings'] - base_bearings) * drag_torque_values['Planetary carrier and bearings']
    drag_oil_pumps = (values_input['N_oil_pumps'] - base_mechanical_oil_pumps) * drag_torque_values['Mechanical oil pump']

    base_drag_torque = 68.11 * math.exp(-25.71 / total_ratio) + 11.28 * math.exp(-0.6 / total_ratio)
    scaling_factor = 1 + 0.12 * (values_input['speeds'] - base_speeds)
    total_drag_torque = (scaling_factor * base_drag_torque) + drag_open_friction_elements + drag_open_dog_clutches + drag_open_synchronisation_gaps + drag_differential_cages + drag_planetary_carriers + drag_bearings + drag_oil_pumps

    RPM_range = values_input['RPM_range']
    output_torque = int(values_input['output_torque'])
    ang_vel = get_ang_vel(np.arange(RPM_range))
    power_loss = total_drag_torque * ang_vel
    total_power_losses = np.tile(power_loss, (output_torque, 1)) / 1000
    total_torque_losses = np.tile(total_drag_torque, (output_torque, RPM_range))

    return total_torque_losses, total_power_losses

def get_ang_vel(RPM):
    """
    Convert RPM to angular velocity in radians per second (rad/s).

    Parameters:
    RPM (np.ndarray): An array of values representing revolutions per minute (RPM).

    Returns:
    np.ndarray: The corresponding angular velocities in radians per second (rad/s).
    """
    return (RPM * 2 * math.pi / 60)

def get_torque(power, ang_vel):
    """
    Calculate torque from power and angular velocity.

    Parameters:
    power (np.ndarray): An array of power values in watts (W).
    ang_vel (np.ndarray): An array of angular velocities in radians per second (rad/s).

    Returns:
    np.ndarray: The corresponding torque values in Newton meters (Nm).
    """
    return (power / ang_vel)

def load_dependent_losses(values_input):
    """
    Calculate load-dependent losses for the gearbox.

    Parameters:
    values_input (dict): A dictionary containing input values needed for the calculation.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Two numpy arrays containing total torque losses and total power losses.
    """
    efficiency_12 = 0.985 
    RPM_range = values_input['RPM_range']
    output_torque = int(values_input['output_torque'])
    total_torque_losses = np.zeros((output_torque, RPM_range), dtype=np.float64)
    total_power_losses = np.zeros((output_torque, RPM_range), dtype=np.float64)

    torques = np.arange(1, output_torque)
    rpms = np.arange(1, RPM_range)
    ang_vel_stages = get_ang_vel(rpms[:, np.newaxis])
    efficiencies = []

    for properties in values_input['gear_stages']:
        if properties.realisation == 'PGS':
            print(f"Using real operation ratio for PGS stage: {properties.real_operation_ratio}")
        ratio = properties.real_operation_ratio if properties.realisation == 'PGS' else properties.ratio
        stage_type = properties.realisation
        operation = properties.operation
        efficiency = calculate_efficiency(stage_type, operation, ratio, efficiency_12, properties.gear_type)
        efficiencies.append(efficiency)

    for torque in torques:
        output_power = torque * ang_vel_stages
        tot_power_loss = np.zeros_like(rpms, dtype=np.float64)
        tot_torque_loss = np.zeros_like(rpms, dtype=np.float64)
        output_torque_stage = torque

        for properties, efficiency in zip(reversed(values_input['gear_stages']), reversed(efficiencies)):
            ratio = properties.real_operation_ratio if properties.realisation == 'PGS' else properties.ratio
            input_power = output_power / efficiency
            input_torque = output_torque_stage / (ratio * efficiency)
            power_loss = input_power * (1 - efficiency)
            torque_loss = input_torque * (1 - efficiency) * ratio

            output_power = input_power
            output_torque_stage = input_torque
            tot_power_loss += power_loss.flatten()
            tot_torque_loss += torque_loss.flatten()

        total_torque_losses[torque, 1:] = tot_torque_loss
        total_power_losses[torque, 1:] = tot_power_loss / 1000

    return total_torque_losses, total_power_losses

def calculate_efficiency(stage_type, operation, ratio, efficiency_12, gear_type=None):
    """
    Calculate the efficiency for a given stage.

    Parameters:
    stage_type (str): The type of gear stage (SG or PGS).
    operation (str): The operation mode of the planetary gear stage.
    ratio (float): The gear ratio for the stage.
    efficiency_12 (float): A base efficiency value for the calculation.
    gear_type (str, optional): The gear type, default is None.

    Returns:
    float: The calculated efficiency.
    """
    if stage_type == 'SG':
        if gear_type == "helical" or gear_type is None:  # Default to helical if gear_type is None
            return 0.99
        elif gear_type == "bevel":
            return 0.98
        elif gear_type == "hypoid":
            return 0.94
        else:
            raise ValueError(f"Unsupported gear type: {gear_type}")
    
    if operation == '1s':
        if ratio < 0:
            return ((ratio * efficiency_12) - 1) / (ratio - 1)
        elif 0 < ratio < 1:
            return ((ratio / efficiency_12) - 1) / (ratio - 1)
        elif ratio > 1:
            return ((ratio * efficiency_12) - 1) / (ratio - 1)
    elif operation == 's1':
        if ratio < 0:
            return (ratio - 1) / ((ratio / efficiency_12) - 1)
        elif 0 < ratio < 1:
            return (ratio - 1) / ((ratio * efficiency_12) - 1)
        elif ratio > 1:
            return (ratio - 1) / ((ratio / efficiency_12) - 1)
    elif operation == '2s':
        if ratio < 0:
            return (ratio - efficiency_12) / (ratio - 1)
        elif 0 < ratio < 1:
            return (ratio - efficiency_12) / (ratio - 1)
        elif ratio > 1:
            return (ratio - (1 / efficiency_12)) / (ratio - 1)
    elif operation == 's2':
        if ratio < 0:
            return (ratio - 1) / (ratio - (1 / efficiency_12))
        elif 0 < ratio < 1:
            return (ratio - 1) / (ratio - (1 / efficiency_12))
        elif ratio > 1:
            return (ratio - 1) / (ratio - efficiency_12)
    elif operation == '12':
        return efficiency_12
    return 0.99

def display_gearbox_stages(gearbox_stages: List[dict], speed: int = None, ratio: float = None):
    """
    Display the calculated gearbox stages with dynamic adaptation to the number of stages.

    Parameters:
    gearbox_stages (List[dict]): A list of dictionaries representing the gearbox stages.
    speed (int, optional): Speed index to display, default is None.
    ratio (float, optional): The gear shift ratio to display, default is None.

    Returns:
    None
    """
    current_stage = None
    stage_number = 0

    for stage in gearbox_stages:
        if stage['stage'] != current_stage:
            stage_number += 1
            current_stage = stage['stage']
            if stage['gear_type'] == 'PGS':
                print(f"\nStage {stage_number} (Planetary Gear):")
            else:
                print(f"\nStage {stage_number} (Simple Gear):")

        if stage['gear_type'] == 'PGS':
            if speed is not None and ratio is not None:
                print(f"  Speed {speed}, Shift Ratio {ratio:.1f}:")
            print(f"    Sun Diameter: {stage['sun_diameter']:.0f} mm")
            print(f"    Sun Teeth: {stage['sun_teeth']}")
            print(f"    Planet Diameter: {stage['planet_diameter']:.0f} mm")
            print(f"    Planet Teeth: {stage['planet_teeth']}")
            print(f"    Ring Diameter: {stage['ring_diameter']:.0f} mm")
            print(f"    Ring Teeth: {stage['ring_teeth']}")
            print(f"    Module: {stage['module']:.2f} mm")
        else:
            print(f"  Driving Gear Diameter: {stage['driving_diameter']:.0f} mm")
            print(f"  Driving Gear Teeth: {stage['driving_teeth']}")
            print(f"  Driven Gear Diameter: {stage['driven_diameter']:.0f} mm")
            print(f"  Driven Gear Teeth: {stage['driven_teeth']}")
            print(f"  Module: {stage['module']:.2f} mm")


def plot_heatmap(data: np.ndarray, x_label: str, y_label: str, title: str, cbar_label: str):
    """
    Plot a heatmap of the provided data.

    Parameters:
    data (np.ndarray): A numpy array containing the data to be plotted.
    x_label (str): Label for the x-axis.
    y_label (str): Label for the y-axis.
    title (str): Title of the plot.
    cbar_label (str): Label for the color bar.

    Returns:
    None
    """
    plt.figure(figsize=(10, 8))
    cbar = plt.colorbar(plt.imshow(data, cmap='viridis', aspect='auto', origin='lower', extent=[0, data.shape[1], 0, data.shape[0]]))
    cbar.set_label(cbar_label)
    ticks = np.linspace(data.min(), data.max(), num=10)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{tick:.0f}' for tick in ticks])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

def calculate_and_display_losses(values_input: dict, speed_index: int, total_ratio: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate and display the losses for the gearbox.

    Parameters:
    values_input (dict): A dictionary of input values for the gearbox.
    speed_index (int): The index of the current speed.
    total_ratio (float): The total gear ratio for the gearbox.

    Returns:
    Tuple[np.ndarray, np.ndarray, float]: Torque losses, power losses, and worst-case input torque.
    """
    independent_torque_losses, independent_power_losses = load_independent_losses(values_input, total_ratio)
    load_dependent_torque_losses, load_dependent_power_losses = load_dependent_losses(values_input)
    total_torque_losses = load_dependent_torque_losses + independent_torque_losses
    total_power_losses = load_dependent_power_losses + independent_power_losses

    max_torque_loss = np.max(total_torque_losses)
    min_torque_loss = np.min(total_torque_losses)

    # Calculate the worst-case input torque
    output_torque = values_input['output_torque']   # Output torque from input values
    worst_case_input_torque = (output_torque + max_torque_loss) / total_ratio

    print(f"Worst case input torque needed from the motor: {worst_case_input_torque:.2f} Nm")
    print(f"Max total torque loss: {max_torque_loss} Nm")
    print(f"Min total torque loss: {min_torque_loss} Nm")

    plot_heatmap(total_torque_losses, 'RPM', 'Torque (Nm)', f'Heatmap of Total Torque Losses (Speed {speed_index + 1})', 'Total Torque Loss (Nm)')

    max_power_loss = np.max(total_power_losses)
    min_power_loss = np.min(total_power_losses)
    print(f"Max total power loss: {max_power_loss} KW")
    print(f"Min total power loss: {min_power_loss} KW")

    plot_heatmap(total_power_losses, 'RPM', 'Torque (Nm)', f'Heatmap of Load Dependent Power Losses (Speed {speed_index + 1})', 'Total Power Loss (KW)')

    return total_torque_losses, total_power_losses, worst_case_input_torque

def round_sizing_values(stage):
    """
    Round the relevant sizing values to 0 decimal places.

    Parameters:
    stage (dict): A dictionary representing the gear stage.

    Returns:
    dict: The updated stage dictionary with rounded values.
    """
    if 'sun_diameter' in stage and stage['sun_diameter'] is not None:
        stage['sun_diameter'] = round(stage['sun_diameter'], 0)
    if 'planet_diameter' in stage and stage['planet_diameter'] is not None:
        stage['planet_diameter'] = round(stage['planet_diameter'], 0)
    if 'ring_diameter' in stage and stage['ring_diameter'] is not None:
        stage['ring_diameter'] = round(stage['ring_diameter'], 0)
    if 'driving_diameter' in stage and stage['driving_diameter'] is not None:
        stage['driving_diameter'] = round(stage['driving_diameter'], 0)
    if 'driven_diameter' in stage and stage['driven_diameter'] is not None:
        stage['driven_diameter'] = round(stage['driven_diameter'], 0)
    return stage

def display_final_sizing_results(final_gear_sizing, most_demanding_stages):
    """
    Display the final sizing results with correct structure.

    Parameters:
    final_gear_sizing (dict): A dictionary containing the final sizing for each stage type.
    most_demanding_stages (dict): A dictionary containing the most demanding stages across configurations.

    Returns:
    None
    """
    print("\nFinal Sizing Results:")

    # Display the most demanding stages that are constant across configurations
    for stage_key, stage in most_demanding_stages.items():
        if stage['gear_type'] == 'SG':
            print(f"\nStage {stage['stage']} (Simple Gear):")
            print(f"  Driving Gear Diameter: {stage['driving_diameter']:.0f} mm")
            print(f"  Driving Gear Teeth: {stage['driving_teeth']}")
            print(f"  Driven Gear Diameter: {stage['driven_diameter']:.0f} mm")
            print(f"  Driven Gear Teeth: {stage['driven_teeth']}")
            print(f"  Module: {stage['module']:.2f} mm")

    # Display the varying Stage 2 (Planetary Gear) configurations
    for speed, stages in final_gear_sizing.items():
        print(f"\n{speed}:")
        for stage in stages:
            if stage['gear_type'] == 'PGS':
                print(f"\nStage {stage['stage']} (Planetary Gear):")
                print(f"  Sun Diameter: {stage['sun_diameter']:.0f} mm")
                print(f"  Sun Teeth: {stage['sun_teeth']}")
                print(f"  Planet Diameter: {stage['planet_diameter']:.0f} mm")
                print(f"  Planet Teeth: {stage['planet_teeth']}")
                print(f"  Ring Diameter: {stage['ring_diameter']:.0f} mm")
                print(f"  Ring Teeth: {stage['ring_teeth']}")
                print(f"  Module: {stage['module']:.2f} mm")



def main():
    global max_ratio_stage_sizes
    max_ratio_stage_sizes = {}

    configuration_file_path = 'Gearbox_Configuration_file.json'
    try:
        values_input = load_configuration(configuration_file_path)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    output_torque = values_input['output_torque']
    material_strength = values_input['material_strength']
    security_factor = values_input['security_factor']
    gear_stages = values_input['gear_stages']

    shift_stage = None
    smallest_shift_ratio = float('inf')

    # Identify the shift stage and validate ratios
    for stage in gear_stages:
        if stage.type == "shift":  # Identify the shift stage by its type
            if not stage.ratios:
                raise ValueError(f"Shift stage '{stage}' is missing 'ratios'. Please provide valid ratios in the configuration.")
            shift_stage = stage
            smallest_shift_ratio = min(stage.ratios)
        elif stage.ratio is None:
            raise ValueError(f"Ratio for stage of type '{stage.type}' is not set. Please check the configuration.")

    # Dictionaries to hold the final sizing for each stage type
    final_gear_sizing = {}
    most_demanding_stages = {}

    if shift_stage:  # If there is a shift stage
        for i, ratio in enumerate(shift_stage.ratios):
            print(f"Running calculations for ratio {ratio} (Speed {i + 1})")
            shift_stage.ratio = ratio  # Temporarily assign the current ratio for calculations

            # Reset total ratio for each shift stage
            total_ratio = 1

            # Size the multistage gearbox
            gearbox_stages = size_multistage_gearbox(output_torque, gear_stages, material_strength, security_factor, smallest_shift_ratio)

            # Calculate the total ratio dynamically
            for gear_stage in gear_stages:
                if gear_stage.realisation == 'SG':
                    total_ratio *= gear_stage.ratio
                elif gear_stage.realisation == 'PGS' and gear_stage.real_operation_ratio is not None:
                    total_ratio *= gear_stage.real_operation_ratio

            # Store the sizes for this shift ratio
            final_gear_sizing[f"Speed {i + 1}, Shift Ratio {ratio}"] = gearbox_stages

            # Track the most demanding values for each type of stage
            for stage in gearbox_stages:
                stage_key = f"stage_{stage['stage']}_{stage['gear_type']}"
                if stage_key not in most_demanding_stages:
                    most_demanding_stages[stage_key] = stage
                else:
                    if stage['gear_type'] == 'SG' and stage['driven_diameter'] > most_demanding_stages[stage_key]['driven_diameter']:
                        most_demanding_stages[stage_key] = stage

            display_gearbox_stages(gearbox_stages, speed=i + 1, ratio=ratio)
            calculate_and_display_losses(values_input, i, total_ratio)

        # Display the final results
        display_final_sizing_results(final_gear_sizing, most_demanding_stages)

    else:  # No shift stage, just process the stages as they are
        total_ratio = 1  # Initialize total ratio for non-shift stages
        gearbox_stages = size_multistage_gearbox(output_torque, gear_stages, material_strength, security_factor, smallest_shift_ratio)

        # Calculate total ratio for non-shift stages
        for gear_stage in gear_stages:
            if gear_stage.realisation == 'SG':
                total_ratio *= gear_stage.ratio
            elif gear_stage.realisation == 'PGS' and gear_stage.real_operation_ratio is not None:
                total_ratio *= gear_stage.real_operation_ratio

        display_gearbox_stages(gearbox_stages)
        calculate_and_display_losses(values_input, 0, total_ratio)

        print("\nFinal Sizing Results:")
        display_gearbox_stages(gearbox_stages)

    plt.show()  # Show all plots at once after all calculations are done



if __name__ == "__main__":
    main()
