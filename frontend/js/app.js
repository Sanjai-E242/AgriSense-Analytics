/**
 * ==========================================================================
 * Crop Recommendation System - Application Bootstrap & Settings Controller
 * ==========================================================================
 */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Render all SVG icons across the shell
  if (window.AppIcons) {
    window.AppIcons.render();
  }

  // 2. Initialize Navigation System
  const navigation = new window.AppNavigation();

  // 3. System Time / Date updater in top bar
  const timeDisplay = document.getElementById('system-time-display');
  function updateTime() {
    if (timeDisplay) {
      const now = new Date();
      timeDisplay.textContent = now.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
      });
    }
  }
  updateTime();

  // 4. Accessible Settings Toggles (Reduced Motion & High Contrast)
  const reducedMotionToggle = document.getElementById('toggle-reduced-motion');
  if (reducedMotionToggle) {
    reducedMotionToggle.addEventListener('change', (e) => {
      if (e.target.checked) {
        document.documentElement.classList.add('force-reduced-motion');
      } else {
        document.documentElement.classList.remove('force-reduced-motion');
      }
    });
  }

  // 5. Console initialization message for verification
  console.info('🌱 Crop Recommendation System: Phase 1 Shell Initialized Successfully.');
});
