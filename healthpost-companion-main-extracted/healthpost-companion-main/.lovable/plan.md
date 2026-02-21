
## Make Medical Image and Current Medications Cards Equal Size

### Problem
The "Add Medical Image" and "Current Medications" accordion cards appear at different heights because they are independent `Accordion` components inside a CSS grid without equal-height enforcement.

### Solution
Apply `h-full` to the inner `AccordionItem` elements (which have the `panel` class) so both cards stretch to fill the grid row height equally. This ensures the two side-by-side cards always match in size regardless of their content.

### Changes

**File: `src/components/ClinicalWorkspace.tsx`**
- On the grid container (`grid grid-cols-1 md:grid-cols-2 gap-4`), add `items-stretch` (default for grid, but explicit for clarity).
- On each `Accordion` wrapper, add `h-full` so the accordion fills the grid cell.
- On each `AccordionItem` (which carries the `panel` class), add `h-full` so the panel card stretches to match its sibling.

This is a CSS-only fix -- no logic or structural changes needed.
