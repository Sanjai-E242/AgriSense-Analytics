/**
 * ==========================================================================
 * Crop Recommendation System - Shell Navigation & Interactive Behavior
 * ==========================================================================
 */

class AppNavigation {
  constructor() {
    this.sidebar = document.getElementById('app-sidebar');
    this.collapseBtn = document.getElementById('sidebar-collapse-btn');
    this.mobileToggle = document.getElementById('mobile-menu-toggle');
    this.overlay = document.getElementById('mobile-overlay');
    this.navLinks = document.querySelectorAll('.nav-link');
    this.breadcrumbCurrent = document.getElementById('breadcrumb-current');
    this.pageTitle = document.getElementById('page-title');
    this.viewSections = document.querySelectorAll('[data-view-panel]');
    this.liveAnnouncer = document.getElementById('aria-live-announcer');

    this.isCollapsed = false;
    this.isMobileOpen = false;

    this.init();
  }

  init() {
    // 1. Sidebar Collapse Listener
    if (this.collapseBtn) {
      this.collapseBtn.addEventListener('click', () => this.toggleSidebarCollapse());
    }

    // 2. Mobile Drawer Listeners
    if (this.mobileToggle) {
      this.mobileToggle.addEventListener('click', () => this.toggleMobileMenu());
    }

    if (this.overlay) {
      this.overlay.addEventListener('click', () => this.closeMobileMenu());
    }

    // 3. Navigation Links (Tab switching between agricultural sections)
    this.navLinks.forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetView = link.getAttribute('data-view');
        const viewTitle = link.getAttribute('data-title');
        this.setActiveView(targetView, viewTitle, link);

        if (this.isMobileOpen) {
          this.closeMobileMenu();
        }
      });
    });

    // 4. Delegated Action Navigation for all buttons/links with [data-navigate-to]
    document.addEventListener('click', (e) => {
      const navTarget = e.target.closest('[data-navigate-to]');
      if (navTarget) {
        e.preventDefault();
        const targetView = navTarget.getAttribute('data-navigate-to');
        const targetLink = document.querySelector(`.nav-link[data-view="${targetView}"]`);
        const viewTitle = targetLink ? targetLink.getAttribute('data-title') : targetView;
        this.setActiveView(targetView, viewTitle, targetLink);
        window.scrollTo({ top: 0, behavior: 'smooth' });

        if (this.isMobileOpen) {
          this.closeMobileMenu();
        }
      }
    });

    // 5. Keyboard Navigation (Escape key, ARIA management)
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.isMobileOpen) {
        this.closeMobileMenu();
      }
    });

    // 6. Restore or initialize layout state & hash route
    this.updateCollapseUI();
    this.initHashRoute();
  }

  initHashRoute() {
    const hash = window.location.hash.replace('#', '');
    if (hash) {
      const targetLink = document.querySelector(`.nav-link[data-view="${hash}"]`);
      if (targetLink) {
        const viewTitle = targetLink.getAttribute('data-title');
        this.setActiveView(hash, viewTitle, targetLink);
      }
    }
  }

  toggleSidebarCollapse() {
    this.isCollapsed = !this.isCollapsed;
    this.updateCollapseUI();
  }

  updateCollapseUI() {
    if (!this.sidebar) return;

    if (this.isCollapsed) {
      this.sidebar.classList.add('collapsed');
      this.collapseBtn.setAttribute('aria-expanded', 'false');
      this.collapseBtn.setAttribute('title', 'Expand Sidebar');
      const iconEl = this.collapseBtn.querySelector('[data-icon]');
      if (iconEl && window.AppIcons) {
        iconEl.innerHTML = window.AppIcons.get('chevronRight');
      }
      const labelEl = this.collapseBtn.querySelector('.collapse-label');
      if (labelEl) labelEl.textContent = 'Expand';
    } else {
      this.sidebar.classList.remove('collapsed');
      this.collapseBtn.setAttribute('aria-expanded', 'true');
      this.collapseBtn.setAttribute('title', 'Collapse Sidebar');
      const iconEl = this.collapseBtn.querySelector('[data-icon]');
      if (iconEl && window.AppIcons) {
        iconEl.innerHTML = window.AppIcons.get('chevronLeft');
      }
      const labelEl = this.collapseBtn.querySelector('.collapse-label');
      if (labelEl) labelEl.textContent = 'Collapse';
    }
  }

  toggleMobileMenu() {
    if (this.isMobileOpen) {
      this.closeMobileMenu();
    } else {
      this.openMobileMenu();
    }
  }

  openMobileMenu() {
    this.isMobileOpen = true;
    this.sidebar.classList.add('mobile-open');
    this.overlay.classList.add('active');
    this.mobileToggle.setAttribute('aria-expanded', 'true');
    const iconEl = this.mobileToggle.querySelector('[data-icon]');
    if (iconEl && window.AppIcons) {
      iconEl.innerHTML = window.AppIcons.get('close');
    }
    this.announce('Mobile navigation opened');
  }

  closeMobileMenu() {
    this.isMobileOpen = false;
    this.sidebar.classList.remove('mobile-open');
    this.overlay.classList.remove('active');
    this.mobileToggle.setAttribute('aria-expanded', 'false');
    const iconEl = this.mobileToggle.querySelector('[data-icon]');
    if (iconEl && window.AppIcons) {
      iconEl.innerHTML = window.AppIcons.get('menu');
    }
    this.announce('Mobile navigation closed');
  }

  setActiveView(viewId, title, activeLinkEl) {
    // 1. Update navigation active state
    this.navLinks.forEach(link => {
      link.classList.remove('active');
      link.removeAttribute('aria-current');
    });

    if (activeLinkEl) {
      activeLinkEl.classList.add('active');
      activeLinkEl.setAttribute('aria-current', 'page');
    }

    // 2. Update Breadcrumb & Document Title
    if (this.breadcrumbCurrent) {
      this.breadcrumbCurrent.textContent = title;
    }
    document.title = `${title} | Crop Recommendation System`;

    // 3. Switch View Panels
    this.viewSections.forEach(section => {
      if (section.getAttribute('data-view-panel') === viewId) {
        section.style.display = 'block';
        section.setAttribute('aria-hidden', 'false');
      } else {
        section.style.display = 'none';
        section.setAttribute('aria-hidden', 'true');
      }
    });

    // 4. Panel-Specific State Synchronization
    if (viewId === 'prediction' && window.predictionForm && typeof window.predictionForm.checkModelStatus === 'function') {
      window.predictionForm.checkModelStatus();
    } else if (viewId === 'history' && window.predictionHistory && typeof window.predictionHistory.loadHistory === 'function') {
      window.predictionHistory.loadHistory();
    } else if (viewId === 'evaluation' && window.modelEvaluation && typeof window.modelEvaluation.checkInitialModelStatus === 'function') {
      window.modelEvaluation.checkInitialModelStatus();
    }

    // 5. Accessibility Announcer
    this.announce(`Navigated to ${title} section`);
  }

  announce(message) {
    if (this.liveAnnouncer) {
      this.liveAnnouncer.textContent = message;
    }
  }
}

// Initialize on DOMContentLoaded
window.AppNavigation = AppNavigation;
