"""
Simple pixel extraction utilities for ProcGen CoinRun
Easy-to-use functions that can be called directly by LLM-generated reward functions
"""
import numpy as np

def get_player_x_position(obs):
    """
    Extract player's x-coordinate from observation
    
    Args:
        obs: (64, 64, 3) numpy array RGB observation
        
    Returns:
        float: Player x position (0-64), or 32.0 if not found
    """
    if obs is None or obs.shape != (64, 64, 3):
        return 32.0
    
    # Strategy 1: Look for blue-ish pixels (common player color)
    blue_mask = (obs[:, :, 2] > obs[:, :, 0] + 30) & (obs[:, :, 2] > obs[:, :, 1] + 30)
    
    # Strategy 2: Look for cyan-ish pixels
    cyan_mask = (obs[:, :, 2] > 100) & (obs[:, :, 1] > 100) & (obs[:, :, 0] < 100)
    
    # Strategy 3: Look for any distinctive non-background pixels in lower half
    lower_half = obs[32:, :, :]
    brightness = np.sum(lower_half, axis=2)
    bright_mask = brightness > 300
    not_sky = ~((lower_half[:, :, 0] > 100) & (lower_half[:, :, 1] > 150) & (lower_half[:, :, 2] > 200))
    distinctive_mask = bright_mask & not_sky
    
    # Try strategies in order of preference
    for mask, y_offset in [(blue_mask, 0), (cyan_mask, 0), (distinctive_mask, 32)]:
        if np.any(mask):
            positions = np.where(mask)
            player_x = np.mean(positions[1])
            return float(player_x)
    
    return 32.0  # Default center

def get_player_y_position(obs):
    """
    Extract player's y-coordinate from observation
    
    Args:
        obs: (64, 64, 3) numpy array RGB observation
        
    Returns:
        float: Player y position (0-64), or 48.0 if not found
    """
    if obs is None or obs.shape != (64, 64, 3):
        return 48.0
    
    # Similar strategies as x position but return y coordinate
    blue_mask = (obs[:, :, 2] > obs[:, :, 0] + 30) & (obs[:, :, 2] > obs[:, :, 1] + 30)
    cyan_mask = (obs[:, :, 2] > 100) & (obs[:, :, 1] > 100) & (obs[:, :, 0] < 100)
    
    for mask in [blue_mask, cyan_mask]:
        if np.any(mask):
            positions = np.where(mask)
            player_y = np.mean(positions[0])
            return float(player_y)
    
    return 48.0  # Default lower portion

def count_coins_visible(obs):
    """
    Count approximate number of coins visible in observation
    
    Args:
        obs: (64, 64, 3) numpy array RGB observation
        
    Returns:
        float: Estimated number of coins (0.0 to ~5.0)
    """
    if obs is None or obs.shape != (64, 64, 3):
        return 0.0
    
    # Strategy 1: Yellow/gold colors
    yellow_mask = (obs[:, :, 0] > 180) & (obs[:, :, 1] > 180) & (obs[:, :, 2] < 120)
    
    # Strategy 2: Orange-ish colors
    orange_mask = (obs[:, :, 0] > 200) & (obs[:, :, 1] > 150) & (obs[:, :, 2] < 100)
    
    # Strategy 3: Warm bright colors (yellow-dominant)
    warm_mask = (obs[:, :, 0] > obs[:, :, 2] + 50) & (obs[:, :, 1] > obs[:, :, 2] + 50) & (obs[:, :, 0] > 120)
    
    # Combine all strategies
    coin_mask = yellow_mask | orange_mask | warm_mask
    coin_pixels = np.sum(coin_mask)
    
    # Convert pixel count to coin estimate (each coin ~10-15 pixels)
    estimated_coins = coin_pixels / 12.0
    return min(estimated_coins, 5.0)  # Cap at reasonable number

def get_obstacle_density(obs):
    """
    Calculate density of obstacles/hazards in observation
    
    Args:
        obs: (64, 64, 3) numpy array RGB observation
        
    Returns:
        float: Obstacle density (0.0 to 1.0)
    """
    if obs is None or obs.shape != (64, 64, 3):
        return 0.0
    
    # Strategy 1: Very dark pixels (saws, enemies)
    dark_mask = np.all(obs < 60, axis=2)
    
    # Strategy 2: Red-ish pixels (danger colors)
    red_mask = (obs[:, :, 0] > 150) & (obs[:, :, 1] < 100) & (obs[:, :, 2] < 100)
    
    # Strategy 3: Gray obstacles
    gray_mask = (np.abs(obs[:, :, 0] - obs[:, :, 1]) < 30) & (np.abs(obs[:, :, 1] - obs[:, :, 2]) < 30) & (obs[:, :, 0] < 120)
    
    obstacle_mask = dark_mask | red_mask | gray_mask
    obstacle_pixels = np.sum(obstacle_mask)
    total_pixels = obs.shape[0] * obs.shape[1]
    
    return float(obstacle_pixels) / total_pixels

def estimate_progress(obs):
    """
    Estimate overall game progress based on visual cues
    
    Args:
        obs: (64, 64, 3) numpy array RGB observation
        
    Returns:
        float: Progress score (0.0 to 10.0)
    """
    if obs is None or obs.shape != (64, 64, 3):
        return 0.0
    
    player_x = get_player_x_position(obs)
    coin_count = count_coins_visible(obs)
    obstacle_density = get_obstacle_density(obs)
    
    # Progress based on rightward movement (0-5 points)
    position_score = (player_x / 64.0) * 5.0
    
    # Bonus for coins (0-2.5 points)
    coin_score = coin_count * 0.5
    
    # Penalty for being in dangerous areas (-2 to 0 points)
    danger_penalty = obstacle_density * 2.0
    
    total_progress = position_score + coin_score - danger_penalty
    return max(0.0, min(10.0, total_progress))

def get_ground_level(obs):
    """
    Estimate the ground/platform level in the observation
    
    Args:
        obs: (64, 64, 3) numpy array RGB observation
        
    Returns:
        float: Ground y-coordinate (higher values = lower on screen)
    """
    if obs is None or obs.shape != (64, 64, 3):
        return 50.0
    
    # Look for brown/tan platform colors
    brown_mask = (obs[:, :, 0] > 80) & (obs[:, :, 0] < 160) & \
                 (obs[:, :, 1] > 60) & (obs[:, :, 1] < 140) & \
                 (obs[:, :, 2] > 40) & (obs[:, :, 2] < 120)
    
    if np.any(brown_mask):
        ground_positions = np.where(brown_mask)
        return float(np.mean(ground_positions[0]))
    
    return 50.0  # Default lower portion

def is_player_on_ground(obs):
    """
    Check if player appears to be on the ground/platform
    
    Args:
        obs: (64, 64, 3) numpy array RGB observation
        
    Returns:
        bool: True if player appears to be on ground
    """
    if obs is None or obs.shape != (64, 64, 3):
        return True
    
    player_y = get_player_y_position(obs)
    ground_y = get_ground_level(obs)
    
    # Player is "on ground" if within 8 pixels of ground level
    return abs(player_y - ground_y) < 8.0

def get_movement_direction(obs_current, obs_previous=None):
    """
    Estimate movement direction by comparing two frames
    
    Args:
        obs_current: (64, 64, 3) current observation
        obs_previous: (64, 64, 3) previous observation (optional)
        
    Returns:
        float: Movement direction (-1.0 = left, 0.0 = stationary, +1.0 = right)
    """
    if obs_previous is None:
        return 0.0
    
    if obs_current is None or obs_current.shape != (64, 64, 3):
        return 0.0
    
    if obs_previous is None or obs_previous.shape != (64, 64, 3):
        return 0.0
    
    current_x = get_player_x_position(obs_current)
    previous_x = get_player_x_position(obs_previous)
    
    movement = current_x - previous_x
    
    # Normalize to -1.0 to +1.0 range
    if movement > 2.0:
        return 1.0  # Moving right
    elif movement < -2.0:
        return -1.0  # Moving left
    else:
        return movement / 2.0  # Small movements

def calculate_pixel_diversity(obs):
    """
    Calculate visual diversity/complexity in observation
    Higher values might indicate more interesting/active areas
    
    Args:
        obs: (64, 64, 3) numpy array RGB observation
        
    Returns:
        float: Diversity score (0.0 to 1.0)
    """
    if obs is None or obs.shape != (64, 64, 3):
        return 0.0
    
    # Count unique colors
    unique_colors = len(np.unique(obs.reshape(-1, 3), axis=0))
    max_possible_colors = min(obs.size // 3, 1000)  # Reasonable maximum
    
    diversity = unique_colors / max_possible_colors
    return min(diversity, 1.0)

# Helper function to combine multiple metrics
def get_comprehensive_state(obs, obs_previous=None):
    """
    Get comprehensive game state information
    
    Args:
        obs: (64, 64, 3) current observation
        obs_previous: (64, 64, 3) previous observation (optional)
        
    Returns:
        dict: Comprehensive state information
    """
    return {
        'player_x': get_player_x_position(obs),
        'player_y': get_player_y_position(obs),
        'coin_count': count_coins_visible(obs),
        'obstacle_density': get_obstacle_density(obs),
        'progress_score': estimate_progress(obs),
        'ground_level': get_ground_level(obs),
        'on_ground': is_player_on_ground(obs),
        'movement_direction': get_movement_direction(obs, obs_previous),
        'visual_diversity': calculate_pixel_diversity(obs)
    }