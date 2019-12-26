## Summary
It's a mini module in Python for simple image generation.

Given a bunch of cropped items with their masks, generate an image with a composition of these items. The generation involves the following procedures:  
1. Random sampling of items from a pool
2. Geometric transformation of the selected items (translation and rotation)
3. Random generation of background (parametrized by strength of noise and constant intensity of background)
4. Change in color contrast of the selected items
5. Random flip in resultant image (horizontally or vertically)


