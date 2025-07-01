#!/usr/bin/env python
"""
Create simple hand icons for left and right indicators
"""

import pygame
import math

def create_hand_icon(direction="left", size=(100, 100), color=(255, 255, 255)):
    """Create a simple hand icon"""
    surface = pygame.Surface(size, pygame.SRCALPHA)
    
    # Hand parameters
    palm_width = int(size[0] * 0.5)
    palm_height = int(size[1] * 0.4)
    finger_width = int(size[0] * 0.12)
    finger_height = int(size[1] * 0.35)
    thumb_width = int(size[0] * 0.15)
    thumb_height = int(size[1] * 0.25)
    
    # Center position
    cx, cy = size[0] // 2, size[1] // 2
    
    if direction == "left":
        # Palm (on the right side for left hand)
        palm_x = cx - palm_width // 4
        palm_y = cy - palm_height // 2
        
        # Draw palm with rounded corners
        pygame.draw.ellipse(surface, color, (palm_x, palm_y, palm_width, palm_height))
        
        # Draw fingers
        finger_spacing = palm_width // 5
        for i in range(4):
            finger_x = palm_x + (i + 0.5) * finger_spacing - finger_width // 2
            finger_y = palm_y - finger_height + 10
            # Make middle fingers slightly longer
            if i == 1 or i == 2:
                finger_y -= 5
            pygame.draw.ellipse(surface, color, 
                              (finger_x, finger_y, finger_width, finger_height))
        
        # Draw thumb (pointing left)
        thumb_x = palm_x - thumb_width + 5
        thumb_y = cy - thumb_height // 2
        pygame.draw.ellipse(surface, color,
                          (thumb_x, thumb_y, thumb_width, thumb_height))
        
        # Add pointing arrow for clarity
        arrow_points = [
            (thumb_x - 10, cy),
            (thumb_x + 5, cy - 8),
            (thumb_x + 5, cy + 8)
        ]
        pygame.draw.polygon(surface, color, arrow_points)
        
    else:  # right
        # Palm (on the left side for right hand)
        palm_x = cx - palm_width // 4
        palm_y = cy - palm_height // 2
        
        # Draw palm with rounded corners
        pygame.draw.ellipse(surface, color, (palm_x, palm_y, palm_width, palm_height))
        
        # Draw fingers
        finger_spacing = palm_width // 5
        for i in range(4):
            finger_x = palm_x + (i + 0.5) * finger_spacing - finger_width // 2
            finger_y = palm_y - finger_height + 10
            # Make middle fingers slightly longer
            if i == 1 or i == 2:
                finger_y -= 5
            pygame.draw.ellipse(surface, color, 
                              (finger_x, finger_y, finger_width, finger_height))
        
        # Draw thumb (pointing right)
        thumb_x = palm_x + palm_width - 5
        thumb_y = cy - thumb_height // 2
        pygame.draw.ellipse(surface, color,
                          (thumb_x, thumb_y, thumb_width, thumb_height))
        
        # Add pointing arrow for clarity
        arrow_points = [
            (thumb_x + thumb_width + 10, cy),
            (thumb_x + thumb_width - 5, cy - 8),
            (thumb_x + thumb_width - 5, cy + 8)
        ]
        pygame.draw.polygon(surface, color, arrow_points)
    
    return surface


def create_simple_hand_icon(direction="left", size=(100, 100), color=(255, 255, 255)):
    """Create a simpler, more stylized hand icon"""
    surface = pygame.Surface(size, pygame.SRCALPHA)
    
    cx, cy = size[0] // 2, size[1] // 2
    
    # Create a simple pointing hand shape
    if direction == "left":
        # Main hand body
        points = [
            (cx + 20, cy - 30),  # Top
            (cx + 20, cy - 10),  # Top right
            (cx - 10, cy - 10),  # Thumb joint
            (cx - 30, cy),       # Thumb tip (pointing left)
            (cx - 10, cy + 10),  # Thumb base
            (cx + 20, cy + 10),  # Bottom right
            (cx + 20, cy + 30),  # Bottom
            (cx, cy + 30),       # Bottom left
            (cx, cy - 30)        # Back to top
        ]
    else:  # right
        # Mirror for right hand
        points = [
            (cx - 20, cy - 30),  # Top
            (cx - 20, cy - 10),  # Top left
            (cx + 10, cy - 10),  # Thumb joint
            (cx + 30, cy),       # Thumb tip (pointing right)
            (cx + 10, cy + 10),  # Thumb base
            (cx - 20, cy + 10),  # Bottom left
            (cx - 20, cy + 30),  # Bottom
            (cx, cy + 30),       # Bottom right
            (cx, cy - 30)        # Back to top
        ]
    
    # Draw filled hand
    pygame.draw.polygon(surface, color, points)
    
    # Add outline for better visibility
    pygame.draw.polygon(surface, (200, 200, 200), points, 2)
    
    # Add directional arrow for extra clarity
    arrow_y = cy
    if direction == "left":
        arrow_points = [
            (cx - 40, arrow_y),
            (cx - 25, arrow_y - 8),
            (cx - 25, arrow_y + 8)
        ]
    else:
        arrow_points = [
            (cx + 40, arrow_y),
            (cx + 25, arrow_y - 8),
            (cx + 25, arrow_y + 8)
        ]
    pygame.draw.polygon(surface, color, arrow_points)
    
    return surface


# Test the icons
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((400, 200))
    clock = pygame.time.Clock()
    
    # Create icons
    left_icon = create_simple_hand_icon("left", (100, 100), (100, 150, 255))
    right_icon = create_simple_hand_icon("right", (100, 100), (255, 150, 100))
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill((20, 20, 30))
        
        # Display icons
        screen.blit(left_icon, (50, 50))
        screen.blit(right_icon, (250, 50))
        
        # Labels
        font = pygame.font.Font(None, 24)
        left_text = font.render("LEFT", True, (255, 255, 255))
        right_text = font.render("RIGHT", True, (255, 255, 255))
        screen.blit(left_text, (75, 160))
        screen.blit(right_text, (270, 160))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    print("Icon test completed")