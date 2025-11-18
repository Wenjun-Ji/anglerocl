#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Angle Robustness Evaluation Metrics

This module provides a series of metrics for evaluating image robustness at different horizontal viewing angles.
It focuses on three core metrics:
1. improved Effective Angle Range (EAR)
2. Angle Robustness Score (ARS)
3. Angle Stability Index (ASI)

These metrics can help quantify a model's ability to maintain detection performance under changes in horizontal viewing angle.
"""
import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Union

def calculate_EAR(angles: List[float], confidences: List[float], 
                  threshold: float, angle_step: float = 1) -> Tuple[float, List[Tuple[float, float]]]:
    """
    Calculate the improved Effective Angle Range, considering potentially non-continuous effective detection intervals.
    
    Parameters:
        angles: List of evaluated angles
        confidences: Corresponding detection confidence for each angle
        threshold: Confidence threshold for determining effective detection
        angle_step: Sampling step of angles, used to determine continuity
        
    Returns:
        total_range: Total angular range of all effective intervals
        ranges: List of all effective intervals, each element is (start angle, end angle)
    """
    # Determine which angles meet the threshold requirements
    valid_angle_indices = [i for i, conf in enumerate(confidences) if conf >= threshold]
    
    if not valid_angle_indices:
        return 0, []
    
    # Get valid angles
    valid_angles = [angles[i] for i in valid_angle_indices]
    valid_angles.sort()
    
    # Find all continuous intervals
    ranges = []
    range_start = valid_angles[0]
    prev_angle = valid_angles[0]
    
    for angle in valid_angles[1:]:
        # If the current angle is not continuous with the previous angle (considering sampling step)
        if angle - prev_angle > angle_step * 1.5:  # Allow small errors
            # Save current interval
            ranges.append((range_start, prev_angle))
            # Start new interval
            range_start = angle
        
        prev_angle = angle
    
    # Add the last interval
    ranges.append((range_start, prev_angle))

    # Calculate the total length of all intervals
    total_range = sum(end - start for start, end in ranges)
    
    return total_range, ranges


def calculate_ARS(angles: List[float], confidences: List[float]) -> Dict[str, float]:
    """
    Calculate the Angle Robustness Score using different weighting schemes.
    
    Parameters:
        angles: List of evaluated angles
        confidences: Corresponding detection confidence for each angle
        
    Returns:
        Dictionary containing ARS values under different weighting schemes
    """
    results = {}
    
    # 1. Uniform weights - all angles equally important
    results['ARS_uniform'] = sum(confidences) / len(confidences) if confidences else 0
    
    # 2. Angle-weighted - extreme angles more important
    angle_weights = [abs(angle) for angle in angles]
    sum_weights = sum(angle_weights)
    if sum_weights > 0:
        results['ARS_angle_weighted'] = sum(conf * weight for conf, weight in 
                                         zip(confidences, angle_weights)) / sum_weights
    else:
        results['ARS_angle_weighted'] = 0
    
    # 3. Gaussian distribution weights - common viewing angles more important
    # Assume that angles within ±30° have higher frequency in practical applications
    gaussian_weights = [math.exp(-(angle**2)/(2*30**2)) for angle in angles]
    sum_gaussian = sum(gaussian_weights)
    if sum_gaussian > 0:
        results['ARS_gaussian'] = sum(conf * weight for conf, weight in 
                                    zip(confidences, gaussian_weights)) / sum_gaussian
    else:
        results['ARS_gaussian'] = 0
    
    return results

def calculate_ASI(angles: List[float], confidences: List[float], sensitivity: float = 3.0, window_size: int = 15) -> Dict[str, Any]:
    """
    Calculate the Angle Stability Index with adjustable sensitivity.
    Uses the formula ASI = e^(-sensitivity * CV) for increased sensitivity.
    
    Parameters:
        angles: List of evaluated angles
        confidences: Corresponding detection confidence for each angle
        sensitivity: Sensitivity multiplier (default 5.0, higher values increase sensitivity)
        window_size: Size of sliding window for local stability calculation (default 10 degrees)
        
    Returns:
        Dictionary containing multiple stability metrics
    """
    results = {}
    
    # Ensure there is data
    if not confidences:
        return {
            'ASI': 0,
            'ASI_local': 0,
            'segment_stability': {}
        }
    
    # Basic ASI calculation - global stability with sensitivity factor
    mean_conf = sum(confidences) / len(confidences)
    if mean_conf > 0:
        std_conf = math.sqrt(sum((c - mean_conf) ** 2 for c in confidences) / len(confidences))
        cv = std_conf / mean_conf
        # Apply sensitivity multiplier
        results['ASI'] = math.exp(-sensitivity * cv)
    else:
        results['ASI'] = 0
    
    # Local volatility using sliding window approach
    if len(confidences) > window_size:
        local_cvs = []
        for i in range(len(confidences) - window_size + 1):
            # Extract window of confidences
            window_confs = confidences[i:i+window_size]
            # Calculate statistics for this window
            window_mean = sum(window_confs) / len(window_confs)
            if window_mean > 0:
                window_std = math.sqrt(sum((c - window_mean) ** 2 for c in window_confs) / len(window_confs))
                window_cv = window_std / window_mean
                local_cvs.append(window_cv)
        
        if local_cvs:
            avg_local_cv = sum(local_cvs) / len(local_cvs)
            # Apply sensitivity multiplier
            results['ASI_local'] = math.exp(-sensitivity * avg_local_cv)
        else:
            results['ASI_local'] = 1
    else:
        # If not enough data for sliding window, use the original approach
        if len(confidences) > 1:
            local_changes = []
            for i in range(1, len(confidences)):
                if confidences[i-1] > 0:
                    change_rate = abs(confidences[i] - confidences[i-1]) / confidences[i-1]
                else:
                    change_rate = abs(confidences[i] - confidences[i-1])
                local_changes.append(change_rate)
            
            avg_local_change = sum(local_changes) / len(local_changes)
            results['ASI_local'] = math.exp(-sensitivity * avg_local_change)
        else:
            results['ASI_local'] = 1
    
    # Segment stability - analyze stability within different angle ranges
    angle_ranges = [(-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 90)]
    segment_stability = {}
    
    for start, end in angle_ranges:
        segment_confs = [conf for angle, conf in zip(angles, confidences) 
                       if start <= angle < end]
        
        if segment_confs:
            seg_mean = sum(segment_confs) / len(segment_confs)
            if seg_mean > 0:
                seg_std = math.sqrt(sum((c - seg_mean) ** 2 for c in segment_confs) / len(segment_confs))
                seg_cv = seg_std / seg_mean
                # Apply sensitivity multiplier
                segment_stability[f"{start}_{end}"] = math.exp(-sensitivity * seg_cv)
            else:
                segment_stability[f"{start}_{end}"] = 0
        else:
            segment_stability[f"{start}_{end}"] = None
    
    results['segment_stability'] = segment_stability
    
    return results

# Added after calculate_ASI function

def calculate_AASR(angles: List[float], confidences: List[float], 
                   threshold: float = 0.5) -> Dict[str, Any]:
    """
    Calculate Angular Attack Success Rate (AASR)
    
    AASR measures the average attack success rate of adversarial samples across all test angles
    at a given confidence threshold. Attack success is defined as: adversarial sample not being 
    detected (conf < threshold)
    
    Parameters:
        angles: List of test angles
        confidences: List of detection confidences for adversarial samples at each angle
        threshold: Detection threshold, default 0.5
        
    Returns:
        Dictionary containing AASR metrics
    """
    if not angles or not confidences:
        return {
            "AASR": 0.0,
            "successful_attacks": 0,
            "total_angles": 0,
            "attack_details": []
        }
    
    # Determine if the attack was successful at each angle
    attack_success = []
    attack_details = []
    
    for angle, conf in zip(angles, confidences):
        # If confidence is above threshold, the attack was successful (successfully detected)
        success = 1 if conf >= threshold else 0
        
        attack_success.append(success)
        attack_details.append({
            "angle": angle,
            "confidence": conf,
            "attack_success": success
        })
    
    # Calculate AASR (attack success rate)
    total_angles = len(attack_success)
    successful_attacks = sum(attack_success)
    aasr = (successful_attacks / total_angles * 100) if total_angles > 0 else 0.0
    
    return {
        "AASR": aasr,
        "successful_attacks": successful_attacks,
        "total_angles": total_angles,
        "attack_details": attack_details
    }