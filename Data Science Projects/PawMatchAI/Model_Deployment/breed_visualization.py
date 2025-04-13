import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple
import pandas as pd
from PIL import Image
from dog_database import get_dog_description
from scoring_calculation_system import UserPreferences, calculate_compatibility_score

def create_visualization_tab(dog_breeds, get_dog_description, calculate_compatibility_score, UserPreferences):
    """Create a visualization tab for breed characteristic analysis"""
    
    # Create shared state container
    shared_preferences = gr.State({
        "living_space": "apartment",
        "yard_access": "no_yard",
        "exercise_time": 60,
        "exercise_type": "moderate_activity",
        "grooming_commitment": "medium",
        "experience_level": "beginner",
        "noise_tolerance": "medium",
        "has_children": False,
        "children_age": "school_age",
        "climate": "moderate"
    })
    
    gr.HTML("""
        <div style='
            text-align: center;
            padding: 20px 0;
            margin: 15px 0;
            background: linear-gradient(to right, rgba(66, 153, 225, 0.1), rgba(72, 187, 120, 0.1));
            border-radius: 10px;
        '>
            <p style='
                font-size: 1.2em;
                margin: 0;
                padding: 0 20px;
                line-height: 1.5;
                background: linear-gradient(90deg, #4299e1, #48bb78);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 600;
            '>
                Gain deeper insight into dog breed characteristics through visualization to help you make a more informed choice.
            </p>
        </div>
    """)
    
    with gr.Tabs():
        # Single breed radar chart analysis tab
        with gr.TabItem("Breed Radar Chart Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    # User interface components - Left side
                    breed_choices = [(breed.replace('_', ' '), breed) for breed in sorted(dog_breeds)]
                    
                    breed_dropdown = gr.Dropdown(
                        label="Select Breed",
                        choices=breed_choices,
                        value=breed_choices[0][1] if breed_choices else None,
                        info="Select a breed to view its characteristics radar chart"
                    )
                    
                    with gr.Accordion("User Preferences (Affects Scoring)", open=False):
                        living_space = gr.Radio(
                            label="Living Space",
                            choices=["apartment", "house_small", "house_large"],
                            value="apartment",
                            info="Your residential environment type"
                        )
                        
                        yard_access = gr.Radio(
                            label="Yard Condition",
                            choices=["no_yard", "shared_yard", "private_yard"],
                            value="no_yard",
                            info="Whether you have yard space"
                        )
                        
                        exercise_time = gr.Slider(
                            label="Daily Exercise Time (minutes)",
                            minimum=15,
                            maximum=180,
                            value=60,
                            step=15,
                            info="Daily exercise time you can provide"
                        )
                        
                        exercise_type = gr.Radio(
                            label="Exercise Type",
                            choices=["light_walks", "moderate_activity", "active_training"],
                            value="moderate_activity",
                            info="Your preferred exercise method"
                        )
                        
                        grooming_commitment = gr.Radio(
                            label="Grooming Commitment",
                            choices=["low", "medium", "high"],
                            value="medium",
                            info="Level of grooming care you're willing to provide"
                        )
                        
                        experience_level = gr.Radio(
                            label="Experience Level",
                            choices=["beginner", "intermediate", "advanced"],
                            value="beginner",
                            info="Your level of dog owning experience"
                        )
                        
                        noise_tolerance = gr.Radio(
                            label="Noise Tolerance",
                            choices=["low", "medium", "high"],
                            value="medium",
                            info="Your acceptance level of dog barking"
                        )
                        
                        has_children = gr.Checkbox(
                            label="Have Children",
                            value=False,
                            info="Whether you have children at home"
                        )
                        
                        children_age = gr.Radio(
                            label="Children's Age",
                            choices=["toddler", "school_age", "teenager"],
                            value="school_age",
                            visible=False,
                            info="Age group of children at home"
                        )
                        
                        climate = gr.Radio(
                            label="Climate Environment",
                            choices=["cold", "moderate", "hot"],
                            value="moderate",
                            info="Climate characteristics of your living area"
                        )
                        
                        # Listen for has_children changes to control children_age display
                        has_children.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=has_children,
                            outputs=children_age
                        )
                        
                        # Add function to update shared preferences
                        def update_shared_preferences(*args):
                            return {
                                "living_space": args[0],
                                "yard_access": args[1],
                                "exercise_time": args[2],
                                "exercise_type": args[3],
                                "grooming_commitment": args[4],
                                "experience_level": args[5],
                                "noise_tolerance": args[6],
                                "has_children": args[7],
                                "children_age": args[8],
                                "climate": args[9]
                            }
                        
                        # Monitor preference changes and update shared state
                        all_preferences = [living_space, yard_access, exercise_time, 
                                          exercise_type, grooming_commitment, experience_level, 
                                          noise_tolerance, has_children, children_age, climate]
                        
                        for pref in all_preferences:
                            pref.change(
                                update_shared_preferences,
                                inputs=all_preferences,
                                outputs=shared_preferences
                            )
                    
                    generate_btn = gr.Button("Generate Radar Chart", variant="primary")
                
                with gr.Column(scale=2):
                    # Right display area
                    radar_plot = gr.Plot(label="Breed Characteristics Radar Chart")
                    breed_details = gr.JSON(label="Breed Detailed Information")
            
            # Button click event
            generate_btn.click(
                fn=lambda *args: generate_radar_chart(
                    args[0], create_user_preferences(*args[1:]), 
                    get_dog_description, calculate_compatibility_score
                ),
                inputs=[breed_dropdown, living_space, yard_access, exercise_time, 
                       exercise_type, grooming_commitment, experience_level, 
                       noise_tolerance, has_children, children_age, climate],
                outputs=[radar_plot, breed_details]
            )
        
        # Breed comparison analysis tab - Improved version
        with gr.TabItem("Breed Comparison Analysis"):
            with gr.Row():
                breed1_dropdown = gr.Dropdown(
                    label="Select First Breed",
                    choices=breed_choices,
                    value=breed_choices[0][1] if breed_choices else None
                )
                
                breed2_dropdown = gr.Dropdown(
                    label="Select Second Breed",
                    choices=breed_choices,
                    value=breed_choices[1][1] if len(breed_choices) > 1 else None
                )
            
            with gr.Row():
                use_shared_settings = gr.Checkbox(
                    label="Use Radar Chart Analysis Settings",
                    value=True,
                    info="Check to use the same preferences from the Radar Chart Analysis tab"
                )
            
            # Custom settings container - only visible when not using shared settings
            with gr.Column(visible=False) as custom_settings:
                with gr.Accordion("Custom Preferences", open=True):
                    comp_living_space = gr.Radio(
                        label="Living Space",
                        choices=["apartment", "house_small", "house_large"],
                        value="apartment"
                    )
                    
                    comp_yard_access = gr.Radio(
                        label="Yard Condition",
                        choices=["no_yard", "shared_yard", "private_yard"],
                        value="no_yard"
                    )
                    
                    comp_exercise_time = gr.Slider(
                        label="Daily Exercise Time (minutes)",
                        minimum=15,
                        maximum=180,
                        value=60,
                        step=15
                    )
                    
                    comp_exercise_type = gr.Radio(
                        label="Exercise Type",
                        choices=["light_walks", "moderate_activity", "active_training"],
                        value="moderate_activity"
                    )
            
            # Toggle custom settings visibility based on checkbox
            use_shared_settings.change(
                fn=lambda x: gr.update(visible=not x),
                inputs=use_shared_settings,
                outputs=custom_settings
            )
            
            compare_btn = gr.Button("Compare Breeds", variant="primary")
            comparison_plot = gr.Plot(label="Breed Characteristics Comparison")
            
            # Improved comparison function that handles both shared and custom settings
            def get_comparison_settings(use_shared, shared_prefs, *custom_prefs):
                """
                Select appropriate settings based on user choice
                
                Args:
                    use_shared: Boolean indicating whether to use shared settings
                    shared_prefs: Dictionary of shared preferences
                    custom_prefs: Custom preference values if not using shared
                    
                Returns:
                    UserPreferences object with the selected settings
                """
                if use_shared:
                    # Use settings from Radar Chart tab
                    return create_user_preferences_from_dict(shared_prefs)
                else:
                    # Use custom settings from Comparison tab
                    return create_user_preferences(
                        custom_prefs[0], custom_prefs[1], custom_prefs[2], custom_prefs[3],
                        "medium", "beginner", "medium", False, "school_age", "moderate"
                    )
            
            # Connect the comparison button
            compare_btn.click(
                fn=lambda breed1, breed2, use_shared, shared_prefs, *custom_prefs: generate_comparison_chart(
                    breed1, breed2, 
                    get_comparison_settings(use_shared, shared_prefs, *custom_prefs),
                    get_dog_description, calculate_compatibility_score
                ),
                inputs=[
                    breed1_dropdown, breed2_dropdown, 
                    use_shared_settings, shared_preferences,
                    comp_living_space, comp_yard_access, 
                    comp_exercise_time, comp_exercise_type
                ],
                outputs=comparison_plot
            )

    return None

def create_user_preferences(living_space, yard_access, exercise_time, exercise_type, 
                          grooming_commitment, experience_level, noise_tolerance, 
                          has_children, children_age, climate):
    """
    Create UserPreferences object from UI inputs
    
    Args:
        living_space: Type of living environment
        yard_access: Yard availability
        exercise_time: Minutes of daily exercise
        exercise_type: Type of exercise activity
        grooming_commitment: Level of grooming commitment
        experience_level: Dog owner experience level
        noise_tolerance: Tolerance for barking
        has_children: Whether there are children in the home
        children_age: Age group of children
        climate: Climate type of the living area
        
    Returns:
        UserPreferences object with the specified settings
    """
    return UserPreferences(
        living_space=living_space,
        yard_access=yard_access,
        exercise_time=exercise_time,
        exercise_type=exercise_type,
        grooming_commitment=grooming_commitment,
        experience_level=experience_level,
        time_availability="moderate",  # Default value
        has_children=has_children,
        children_age=children_age if has_children else "school_age",
        noise_tolerance=noise_tolerance,
        space_for_play=True,  # Default value
        other_pets=False,  # Default value
        climate=climate
    )

def create_user_preferences_from_dict(prefs_dict):
    """
    Create UserPreferences object from a dictionary
    
    Args:
        prefs_dict: Dictionary containing preference values
        
    Returns:
        UserPreferences object populated with the dictionary values
    """
    return UserPreferences(
        living_space=prefs_dict["living_space"],
        yard_access=prefs_dict["yard_access"],
        exercise_time=prefs_dict["exercise_time"],
        exercise_type=prefs_dict["exercise_type"],
        grooming_commitment=prefs_dict["grooming_commitment"],
        experience_level=prefs_dict["experience_level"],
        time_availability="moderate",  # Default value
        has_children=prefs_dict["has_children"],
        children_age=prefs_dict["children_age"],
        noise_tolerance=prefs_dict["noise_tolerance"],
        space_for_play=True,  # Default value
        other_pets=False,  # Default value
        climate=prefs_dict["climate"]
    )

def generate_radar_chart(breed_name, user_prefs, get_dog_description, calculate_compatibility_score):
    """
    Generate radar chart for a single breed
    
    Args:
        breed_name: Dog breed name
        user_prefs: UserPreferences object
        get_dog_description: Function to get breed description
        calculate_compatibility_score: Function to calculate compatibility score
    
    Returns:
        tuple: (matplotlib figure, breed description dict)
    """
    try:
        # Get breed description
        breed_info = get_dog_description(breed_name)
        
        if not breed_info:
            # Create empty figure with error message
            fig = Figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"No information found for breed: {breed_name}",
                   horizontalalignment='center', verticalalignment='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.axis('off')
            return fig, {"error": f"No information found for breed: {breed_name}"}
        
        # Calculate compatibility scores
        scores = calculate_compatibility_score(breed_info, user_prefs)
        
        # Prepare data for radar chart
        categories = ['Space Compatibility', 'Exercise Needs', 'Grooming', 
                     'Experience Required', 'Health', 'Noise Level']
        values = [scores['space'], scores['exercise'], scores['grooming'], 
                 scores['experience'], scores['health'], scores['noise']]
        
        # Close the polygon by appending first value
        values_closed = values + [values[0]]
        categories_closed = categories + [categories[0]]
        
        # Calculate angles for each category
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Create figure and polar axis
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Plot data
        ax.fill(angles, values_closed, color='skyblue', alpha=0.25)
        ax.plot(angles, values_closed, color='blue', linewidth=2)
        
        # Add category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        
        # Configure y-axis
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.set_ylim(0, 1)
        
        # Add a title
        breed_display_name = breed_name.replace('_', ' ')
        ax.set_title(f"{breed_display_name} Characteristic Scores", fontsize=16, pad=20)
        
        # Add value labels at each point
        for i, (angle, value) in enumerate(zip(angles[:-1], values)):
            ax.text(angle, value + 0.05, f"{value:.2f}", 
                   ha='center', va='center', fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3"))
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add overall score text
        overall_score = scores.get('overall', 0)
        fig.text(0.5, 0.02, f"Overall Match Score: {overall_score:.2f}", 
                ha='center', fontsize=14, 
                bbox=dict(facecolor='lightgreen', alpha=0.3, boxstyle="round,pad=0.5"))
        
        # Enhance aesthetics
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f0f0f0')
        
        # Print debug information
        print(f"Generated radar chart for {breed_name}")
        print(f"Scores: {scores}")
        
        return fig, breed_info
    
    except Exception as e:
        # Create empty figure with error message
        fig = Figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"Error generating chart: {str(e)}",
               horizontalalignment='center', verticalalignment='center', 
               transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        print(f"Error in generate_radar_chart: {str(e)}")
        return fig, {"error": f"Error generating chart: {str(e)}"}

def generate_comparison_chart(breed1, breed2, user_prefs, get_dog_description, calculate_compatibility_score):
    """
    Generate comparison chart for two breeds
    
    Args:
        breed1, breed2: Dog breed names
        user_prefs: UserPreferences object
        get_dog_description: Function to get breed description
        calculate_compatibility_score: Function to calculate compatibility score
    
    Returns:
        matplotlib figure: Comparison chart
    """
    try:
        # Get breed descriptions
        breed1_info = get_dog_description(breed1)
        breed2_info = get_dog_description(breed2)
        
        if not breed1_info or not breed2_info:
            # Create empty figure with error message
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Missing breed information. Please check both breeds.",
                   horizontalalignment='center', verticalalignment='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.axis('off')
            return fig
        
        # Calculate compatibility scores
        scores1 = calculate_compatibility_score(breed1_info, user_prefs)
        scores2 = calculate_compatibility_score(breed2_info, user_prefs)
        
        # Prepare data for bar chart
        categories = ['Space Compatibility', 'Exercise Needs', 'Grooming', 
                     'Experience Required', 'Health', 'Noise Level']
        values1 = [scores1['space'], scores1['exercise'], scores1['grooming'], 
                  scores1['experience'], scores1['health'], scores1['noise']]
        values2 = [scores2['space'], scores2['exercise'], scores2['grooming'], 
                  scores2['experience'], scores2['health'], scores2['noise']]
        
        # Create figure
        fig = Figure(figsize=(12, 7))
        ax = fig.add_subplot(111)
        
        # Set width of bars
        x = np.arange(len(categories))
        width = 0.35
        
        # Plot bars
        breed1_display = breed1.replace('_', ' ')
        breed2_display = breed2.replace('_', ' ')
        
        rects1 = ax.bar(x - width/2, values1, width, label=breed1_display, color='#4299e1')
        rects2 = ax.bar(x + width/2, values2, width, label=breed2_display, color='#f56565')
        
        # Add labels and title
        ax.set_xlabel('Scoring Dimensions', fontsize=12)
        ax.set_ylabel('Score (0-1)', fontsize=12)
        ax.set_title(f'{breed1_display} vs {breed2_display} Breed Comparison', fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=30, ha='right')
        ax.legend(loc='upper right')
        
        # Add value labels on top of bars
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9, fontweight='bold')
        
        add_labels(rects1)
        add_labels(rects2)
        
        # Set y-axis limit
        ax.set_ylim(0, 1.1)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        # Add overall score comparison
        overall1 = scores1.get('overall', 0)
        overall2 = scores2.get('overall', 0)
        
        fig.text(0.5, 0.02, 
                f"Overall Match Scores:  {breed1_display}: {overall1:.2f}  |  {breed2_display}: {overall2:.2f}", 
                ha='center', fontsize=13, 
                bbox=dict(facecolor='#edf2f7', alpha=0.7, boxstyle="round,pad=0.5"))
        
        # Enhance aesthetics
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f0f0f0')
        
        # Add a tight layout to ensure everything fits
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Print debug information
        print(f"Generated comparison chart for {breed1} vs {breed2}")
        
        return fig
    
    except Exception as e:
        # Create empty figure with error message
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"Error generating comparison: {str(e)}",
               horizontalalignment='center', verticalalignment='center', 
               transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        print(f"Error in generate_comparison_chart: {str(e)}")
        return fig
