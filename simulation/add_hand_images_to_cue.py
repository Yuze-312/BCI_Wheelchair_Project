#!/usr/bin/env python
"""
Script to add hand images to the word cue display in run_simulation_2.py
"""

import os

# The modification to add to the beginning of run_simulation_2.py (after pygame import)
HAND_IMAGE_LOAD_CODE = """
# Load hand images for motor imagery cues
LEFT_HAND_IMG = None
RIGHT_HAND_IMG = None
try:
    LEFT_HAND_IMG = pygame.image.load(os.path.join(os.path.dirname(__file__), 'left_hand.png'))
    RIGHT_HAND_IMG = pygame.image.load(os.path.join(os.path.dirname(__file__), 'right_hand.png'))
    # Scale images to appropriate size (adjust as needed)
    HAND_IMG_SIZE = (80, 80)  # Adjust this size as needed
    LEFT_HAND_IMG = pygame.transform.scale(LEFT_HAND_IMG, HAND_IMG_SIZE)
    RIGHT_HAND_IMG = pygame.transform.scale(RIGHT_HAND_IMG, HAND_IMG_SIZE)
    print("Hand images loaded successfully")
except Exception as e:
    print(f"Warning: Could not load hand images: {e}")
    LEFT_HAND_IMG = None
    RIGHT_HAND_IMG = None
"""

# The modification for the word cue rendering section
HAND_IMAGE_DISPLAY_CODE = """
    # Display hand image alongside word cue
    if word_cue.cue_word == "LEFT" and LEFT_HAND_IMG:
        # Position left hand image to the left of the text
        hand_x = cue_x + 20
        hand_y = cue_y + 20
        screen.blit(LEFT_HAND_IMG, (hand_x, hand_y))
        
        # Add subtle glow effect
        glow_surface = pygame.Surface((HAND_IMG_SIZE[0] + 20, HAND_IMG_SIZE[1] + 20), pygame.SRCALPHA)
        glow_alpha = int(100 * arrow_pulse)
        pygame.draw.circle(glow_surface, (*border_color, glow_alpha), 
                         (HAND_IMG_SIZE[0]//2 + 10, HAND_IMG_SIZE[1]//2 + 10), 
                         HAND_IMG_SIZE[0]//2 + 5)
        screen.blit(glow_surface, (hand_x - 10, hand_y - 10))
        
    elif word_cue.cue_word == "RIGHT" and RIGHT_HAND_IMG:
        # Position right hand image to the right of the text
        hand_x = cue_x + cue_width - HAND_IMG_SIZE[0] - 20
        hand_y = cue_y + 20
        screen.blit(RIGHT_HAND_IMG, (hand_x, hand_y))
        
        # Add subtle glow effect
        glow_surface = pygame.Surface((HAND_IMG_SIZE[0] + 20, HAND_IMG_SIZE[1] + 20), pygame.SRCALPHA)
        glow_alpha = int(100 * arrow_pulse)
        pygame.draw.circle(glow_surface, (*border_color, glow_alpha), 
                         (HAND_IMG_SIZE[0]//2 + 10, HAND_IMG_SIZE[1]//2 + 10), 
                         HAND_IMG_SIZE[0]//2 + 5)
        screen.blit(glow_surface, (hand_x - 10, hand_y - 10))
"""

print("Code snippets to add hand images have been prepared.")
print("\nTo implement:")
print("1. Add HAND_IMAGE_LOAD_CODE after pygame imports in run_simulation_2.py")
print("2. Add HAND_IMAGE_DISPLAY_CODE after line 1052 (after arrow drawing)")
print("\nOr run this script with --apply flag to automatically update the file.")

if "--apply" in os.sys.argv:
    # Read the original file
    with open("run_simulation_2.py", "r") as f:
        content = f.read()
    
    # Find where to insert the image loading code (after pygame import)
    import_pos = content.find("import pygame")
    if import_pos != -1:
        # Find the end of the import section
        next_section = content.find("\n\n", import_pos)
        if next_section != -1:
            # Insert the hand image loading code
            content = content[:next_section] + "\n" + HAND_IMAGE_LOAD_CODE + content[next_section:]
    
    # Find where to insert the display code
    # Look for the arrow drawing section
    arrow_section = content.find("pygame.draw.polygon(screen, (*border_color, int(glow_alpha * arrow_pulse)), arrow_points)")
    if arrow_section != -1:
        # Find the end of the arrow drawing block
        insert_pos = content.find("\n    \n    # Word text", arrow_section)
        if insert_pos != -1:
            content = content[:insert_pos] + "\n" + HAND_IMAGE_DISPLAY_CODE + content[insert_pos:]
    
    # Write the modified content
    with open("run_simulation_2_with_hands.py", "w") as f:
        f.write(content)
    
    print("\nModified file saved as: run_simulation_2_with_hands.py")